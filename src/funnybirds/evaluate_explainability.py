import argparse
import random

import torch
from captum.attr import IntegratedGradients, InputXGradient

from models.resnet import resnet50
from models.vgg import vgg16
from models.ViT.ViT_new import vit_base_patch16_224
from models.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from models.model_wrapper import StandardModel, ViTModel
from evaluation_protocols import accuracy_protocol, controlled_synthetic_data_check_protocol, single_deletion_protocol, preservation_check_protocol, deletion_check_protocol, target_sensitivity_protocol, distractibility_protocol, background_independence_protocol
from explainers.explainer_wrapper import CaptumAttributionExplainer, ViTGradCamExplainer, ViTRolloutExplainer, ViTCheferLRPExplainer, CustomExplainer
from explainers.explainer_wrapper import STEWrapper, AbstractAttributionExplainer, STEAttributionExplainer
from benchmark import ModelEnv


parser = argparse.ArgumentParser(description='FunnyBirds - Explanation Evaluation')
parser.add_argument('--data', metavar='DIR', required=True,
                    help='path to dataset (default: imagenet)')
parser.add_argument('--model', required=True,
                    choices=['resnet50', 'vgg16', 'vit_b_16'],
                    help='model architecture')
parser.add_argument('--explainer', required=True,
                    choices=['IntegratedGradients', 'InputXGradient', 'Rollout', 'CheferLRP', 'CustomExplainer',
                             'xGC', 'xLSC', 'xLC', 'xRISE', 'xEP', 'xAC', 'xGC++', 'xFG', 'xIG', 'xGIG','xDIX','xMP'],
                    help='explainer')
parser.add_argument('--checkpoint_name', type=str, required=False, default=None,
                    help='checkpoint name (including dir)')

parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--seed', default=0, type=int,
                    help='seed')
parser.add_argument('--batch_size', default=32, type=int,
                    help='batch size for protocols that do not require custom BS such as accuracy')
parser.add_argument('--nr_itrs', default=2501, type=int,
                    help='batch size for protocols that do not require custom BS such as accuracy')
                    
parser.add_argument('--accuracy', default=True, action='store_true',
                    help='compute accuracy')
parser.add_argument('--controlled_synthetic_data_check', default=True, action='store_true',
                    help='compute controlled synthetic data check')
parser.add_argument('--single_deletion', default=True, action='store_true',
                    help='compute single deletion')
parser.add_argument('--preservation_check', default=True, action='store_true',
                    help='compute preservation check')
parser.add_argument('--deletion_check', default=True, action='store_true',
                    help='compute deletion check')
parser.add_argument('--target_sensitivity', default=True, action='store_true',
                    help='compute target sensitivity')
parser.add_argument('--distractibility', default=True, action='store_true',
                    help='compute distractibility')
parser.add_argument('--background_independence', default=True, action='store_true',
                    help='compute background dependence')




def main():
    args = parser.parse_args()
    device = 'cuda:' + str(args.gpu)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"### args: {args}")
    # create model
    model_name = args.model
    if args.model == 'resnet50':
        model = resnet50(num_classes = 50)
        model = StandardModel(model)
    elif args.model == 'vgg16':
        model = vgg16(num_classes = 50)
        model = StandardModel(model)
    elif args.model == 'vit_b_16':
        model_name = "vit_base_patch16_224"
        if args.explainer == 'CheferLRP':
            model = vit_LRP(num_classes=50)
        else:
            model = vit_base_patch16_224(num_classes = 50)
        model = ViTModel(model)
    else:
        print('Model not implemented')
    
    if args.checkpoint_name:
        model.load_state_dict(torch.load(args.checkpoint_name, map_location=torch.device('cpu'))['state_dict'])
    model = model.to(device)
    model.eval()

    ######
    me = ModelEnv(model_name)
    me.model = model.model
    me.shape = (256,256)
    #print("=====================")
    #print(model)
    #print("=====================")

    # create explainer
    if args.explainer == 'InputXGradient':
        explainer = InputXGradient(model)
        explainer = CaptumAttributionExplainer(explainer)
    elif args.explainer == 'IntegratedGradients':
        explainer = IntegratedGradients(model)
        baseline = torch.zeros((1,3,256,256)).to(device)
        explainer = CaptumAttributionExplainer(explainer, baseline=baseline)
    elif args.explainer == 'Rollout':
        explainer = ViTRolloutExplainer(model)
    elif args.explainer == 'CheferLRP':
        explainer = ViTCheferLRPExplainer(model)
    elif args.explainer == "xGC":
        from adaptors import CaptumCamSaliencyCreator, CamSaliencyCreator, METHOD_CONV, CMethod
        salc = CamSaliencyCreator([CMethod.GradCAM])
        explainer = STEAttributionExplainer(salc, me)
    elif args.explainer == "xGC++":
        from adaptors import CaptumCamSaliencyCreator, CamSaliencyCreator, METHOD_CONV, CMethod
        salc = CamSaliencyCreator([CMethod.GradCAMPlusPlus])
        explainer = STEAttributionExplainer(salc, me)
    elif args.explainer == "xFG":
        from adaptors import CaptumCamSaliencyCreator, CamSaliencyCreator, METHOD_CONV, CMethod
        salc = CamSaliencyCreator([CMethod.FullGrad])
        explainer = STEAttributionExplainer(salc, me)
    elif args.explainer == "xLC":
        from adaptors import CaptumCamSaliencyCreator, CamSaliencyCreator, METHOD_CONV, CMethod
        salc = CamSaliencyCreator([CMethod.LayerCAM])
        explainer = STEAttributionExplainer(salc, me)
    elif args.explainer == "xAC":
        from adaptors import CaptumCamSaliencyCreator, CamSaliencyCreator, METHOD_CONV, CMethod
        salc = CamSaliencyCreator([CMethod.AblationCAM])
        explainer = STEAttributionExplainer(salc, me)
        
    elif args.explainer == "xLSC":
        from lcpe import CompExpCreator
        salc = CompExpCreator(
            desc="MrCompJ", segsize=[16,48], nmasks=[500,500], c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9, 
            epochs=101, ##select_from=10, select_freq=3, select_del=1.0,
            c_mask_completeness=1.0, c_magnitude=0.01, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, 
            c_activation="")
        explainer = STEAttributionExplainer(salc, me)
    elif args.explainer == "xDIX":
        from dix_cnn import DixCnnSaliencyCreator
        salc = DixCnnSaliencyCreator(alt_model=True)
        explainer = STEAttributionExplainer(salc, me)
    elif args.explainer == "xEP":
        from extpert import ExtPertSaliencyCreator        
        salc = ExtPertSaliencyCreator(single=True)
        explainer = STEAttributionExplainer(salc, me)
    elif args.explainer == "xMP":
        from mpert import IEMPertSaliencyCreator 
        salc = IEMPertSaliencyCreator()
        explainer = STEAttributionExplainer(salc, me)
    elif args.explainer == "xIG":
        from adaptors_gig import IGSaliencyCreator
        salc = IGSaliencyCreator(methods=["IG"])        
        explainer = STEAttributionExplainer(salc, me)
    elif args.explainer == "xGIG":
        from adaptors_gig import IGSaliencyCreator
        salc = IGSaliencyCreator(methods=["GIG"])        
        explainer = STEAttributionExplainer(salc, me)
    elif args.explainer == "xRISE":
        from RISE import RiseSaliencyCreator        
        salc = RiseSaliencyCreator()
        explainer = STEAttributionExplainer(salc, me)

    elif args.explainer == 'CustomExplainer':
        ...
    else:
        print('Explainer not implemented')

    import time
    start_time = time.time()
    accuracy, csdc, pc, dc, distractibility, sd, ts = -1, -1, -1, -1, -1, -1, -1

    if args.accuracy:
        print('Computing accuracy...')
        accuracy = accuracy_protocol(model, args)
        accuracy = round(accuracy, 5)
        print(f'Result - Accuracy: {accuracy}')

    if args.controlled_synthetic_data_check:
        print('Computing controlled synthetic data check...')
        csdc = controlled_synthetic_data_check_protocol(model, explainer, args)
        print(f'Result - CSDC: {csdc}')

    if args.target_sensitivity:
        print('Computing target sensitivity...')
        ts = target_sensitivity_protocol(model, explainer, args)
        ts = round(ts, 5)
        print(f'Result - TS: {ts}')

    if args.single_deletion:
        print('Computing single deletion...')
        sd = single_deletion_protocol(model, explainer, args)
        sd = round(sd, 5)
        print(f'Result - SD: {sd}')

    if args.preservation_check:
        print('Computing preservation check...')
        pc = preservation_check_protocol(model, explainer, args)
        print(f'Result - PC: {pc}')

    if args.deletion_check:
        print('Computing deletion check...')
        dc = deletion_check_protocol(model, explainer, args)
        print(f'Result - DC: {dc}')

    if args.distractibility:
        print('Computing distractibility...')
        distractibility = distractibility_protocol(model, explainer, args)
        print(f'Result - dist: {distractibility}')

    if args.background_independence:
        print('Computing background independence...')
        background_independence = background_independence_protocol(model, args)
        background_independence = round(background_independence, 5)
        print(f'Result - BI: {background_independence}')
    
    # select completeness and distractability thresholds such that they maximize the sum of both
    max_score = 0
    best_threshold = -1
    for threshold in csdc.keys():
        max_score_tmp = csdc[threshold]/3. + pc[threshold]/3. + dc[threshold]/3. + distractibility[threshold]
        if max_score_tmp > max_score:
            max_score = max_score_tmp
            best_threshold = threshold

    print('FINAL RESULTS:')
    print('Accuracy, CSDC, PC, DC, Distractability, Background independence, SD, TS')
    print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(accuracy, round(csdc[best_threshold],5), round(pc[best_threshold],5), round(dc[best_threshold],5), round(distractibility[best_threshold],5), background_independence, sd, ts))
    print('Best threshold:', best_threshold)

    end_time = time.time()
    print(f"Total time: {end_time-start_time}")
if __name__ == '__main__':
    main()