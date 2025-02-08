import torch


from baselines.dix import setup    
from salieny_models import GradModel
    
from saliency_utils import * # type: ignore
from salieny_models import * # type: ignore
from saliency_lib import * # type: ignore
import time
from reports import report_duration


INTERPOLATION_STEPS = 4

def get_grads_wrt_image(model, label, images_batch, device='cuda', steps=50):
    model.eval()
    model.zero_grad()

    images_batch.requires_grad = True
    preds = model(images_batch.to(device), hook=True)
    _, predicted = torch.max(preds.data, 1)
    one_hot = torch.zeros(preds.shape).to(device)
    one_hot[:, label] = 1

    score = torch.sum(one_hot * preds)
    score.backward()
    with torch.no_grad():
        image_grads = images_batch.grad.detach()
    images_batch.requires_grad = False
    return image_grads


def backward_class_score_and_get_activation_grads(model, label, x, only_post_features=False, device='cuda',
                                                  is_middle=False):
    model.zero_grad()

    print("###", x.shape) ## PPP
    preds = model(x.to(device), hook=True, only_post_features=only_post_features,
                  is_middle=is_middle)
    _, predicted = torch.max(preds.data, 1)
    one_hot = torch.zeros(preds.shape).to(device)
    one_hot[:, label] = 1

    score = torch.sum(one_hot * preds)
    score.backward()

    activations_gradients = model.get_activations_gradient().unsqueeze(
        1).detach().cpu()

    return activations_gradients


def backward_class_score_and_get_images_grads(model, label, x, only_post_features=False, device='cuda'):
    model.zero_grad()
    preds = model(x.squeeze(1).to(device), hook=True)
    _, predicted = torch.max(preds.data, 1)
    one_hot = torch.zeros(preds.shape).to(device)
    one_hot[:, label] = 1

    score = torch.sum(one_hot * preds)
    score.backward()

    images_gradients = model.get_activations_gradient().unsqueeze(
        1).detach().cpu()

    return images_gradients


def get_blurred_values(target, num_steps):
    num_steps += 1
    if num_steps <= 0: return np.array([])
    target = target.squeeze()
    tshape = len(target.shape)
    blurred_images_list = []
    for step in range(num_steps):
        sigma = int(step) / int(num_steps)
        sigma_list = [sigma, sigma, 0]

        if tshape == 4:
            sigma_list = [sigma, sigma, sigma, 0]

        blurred_image = ndimage.gaussian_filter(
            target.detach().cpu().numpy(), sigma=sigma_list, mode="grid-constant")
        blurred_images_list.append(blurred_image)

    return numpy.array(blurred_images_list)



def heatmap_of_layer_activations_integration_mulacts(device, inp, interpolation_on_activations_steps_arr,
                                                     interpolation_on_images_steps_arr,
                                                     label, layers, model_name):
    print(layers[0]) ## PUSH_ASSERT
    model = GradModel(model_name, feature_layer=layers[0])
    model.to(device)
    model.eval()
    model.zero_grad()

    label = torch.tensor(label, dtype=torch.long, device=device)
    activations = model.get_activations(inp).cpu()
    print("activations:", activations.shape)
    activations_featmap_list = (activations.unsqueeze(1))

    x, _ = torch.min(activations_featmap_list, dim=1)
    basel = torch.ones_like(activations_featmap_list) * x.unsqueeze(1)
    igacts = get_interpolated_values(basel.detach(), activations_featmap_list,
                                     INTERPOLATION_STEPS).detach()

    grads = []
    for act in igacts:
        act.requires_grad = True

        diff2 = (act - basel) / INTERPOLATION_STEPS
        normalic2 = torch.norm(diff2)

        grads.append(F.relu(calc_grads_model(model, act, device, label).detach()) * F.relu(act))
        act = act.detach()
        act.requires_grad = False

    with torch.no_grad():
        igrads = torch.stack(grads).detach()

        mul_grad_act = (igrads.squeeze().detach())
        gradsum = torch.sum(mul_grad_act, dim=[0])
        integrated_heatmaps = gradsum

    return inp, integrated_heatmaps


def heatmap_of_layer_activations_integration(device, inp, interpolation_on_activations_steps_arr,
                                             interpolation_on_images_steps_arr,
                                             label, layers, model_name):
    print(layers[0])
    model = GradModel(model_name, feature_layer=layers[0])
    model.to(device)
    model.eval()
    model.zero_grad()

    label = torch.tensor(label, dtype=torch.long, device=device)
    activations = model.get_activations(inp).cpu()

    activations_featmap_list = (activations.unsqueeze(1))

    x, _ = torch.min(activations_featmap_list, dim=0)
    basel = torch.ones_like(activations_featmap_list) * x.unsqueeze(0)

    igacts = get_interpolated_values(basel.detach(), activations_featmap_list,
                                     INTERPOLATION_STEPS).detach()

    grads = []
    for act in igacts:
        act.requires_grad = True

        diff2 = (act - basel) / INTERPOLATION_STEPS
        normalic2 = torch.norm(diff2)

        grads.append(calc_grads_model(model, act, device, label).detach() * act)
        act = act.detach()
        act.requires_grad = False

    with torch.no_grad():
        igrads = torch.stack(grads).detach()
        mul_grad_act = F.relu(igrads.squeeze().detach())
        integrated_heatmaps = torch.sum(mul_grad_act, dim=[0])

    return inp, integrated_heatmaps

### PPP
def heatmap_of_layer_before_last_integrand(device, inp, interpolation_on_activations_steps_arr,
                                           interpolation_on_images_steps_arr,
                                           label, layers, model_name):
    print(layers)
    model = GradModel(model_name, feature_layer=layers)
    model.to(device)
    model.eval()
    model.zero_grad()

    label = torch.tensor(label, dtype=torch.long, device=device)

    original_activations = model.get_activations(inp).cpu()
    original_activations_featmap_list = original_activations

    x, _ = torch.min(original_activations_featmap_list, dim=1)
    basel = torch.ones_like(original_activations_featmap_list) * x.unsqueeze(1)

    igacts = get_interpolated_values(basel.detach(), original_activations_featmap_list,
                                     INTERPOLATION_STEPS).detach()

    grads = []
    for act in igacts:
        act.requires_grad = True

        diff2 = (act - basel) / INTERPOLATION_STEPS
        normalic2 = torch.norm(diff2) ## PPP
        print(">>>", act.shape)
        grads.append((calc_grads_model(model, act.unsqueeze(0), device, label).detach()) * F.relu(act) * normalic2)
        act = act.detach()
        act.requires_grad = False

    with torch.no_grad():
        igrads = torch.stack(grads).detach()
        mul_grad_act = F.relu(igrads.squeeze().detach())
        integrated_heatmaps = torch.sum(mul_grad_act, dim=[0])

    return inp, integrated_heatmaps


def heatmap_of_layers_layer_no_interpolation(device, inp, interpolation_on_activations_steps_arr,
                                             interpolation_on_images_steps_arr,
                                             label, layers, model_name):
    model = GradModel(model_name, feature_layer=layers[0])
    model.to(device)
    model.eval()
    model.zero_grad()


    label = torch.tensor(label, dtype=torch.long, device=device)
    activations = model.get_activations(inp).cpu()
    activations_featmap_list = (activations.unsqueeze(1))
    gradients = calc_grads_model(model, activations_featmap_list, device, label).detach()
    gradients_squeeze = gradients.detach().squeeze()
    act_grads = F.relu(activations.squeeze()) * F.relu(gradients_squeeze) ** 2
    integrated_heatmaps = torch.sum(act_grads.squeeze(0), dim=0).unsqueeze(0).unsqueeze(0)
    return inp, integrated_heatmaps


def calc_grads_model(model, activations_featmap_list, device, label):
    ## PPP
    activations_gradients = backward_class_score_and_get_activation_grads(model, label, activations_featmap_list,
                                                                          only_post_features=True,
                                                                          device=device)
    return activations_gradients



def get_by_class_saliency_dix(inp,
                              label,                              
                              model_name='densnet',
                              layers=[12],
                              interpolation_on_images_steps_arr=[0, 50],
                              interpolation_on_activations_steps_arr=[0, 50],
                              device='cuda',
                              use_mask=False):
    print("START")
    images, integrated_heatmaps1 = heatmap_of_layer_activations_integration_mulacts(device, inp,
                                                                                    interpolation_on_activations_steps_arr,
                                                                                    interpolation_on_images_steps_arr,
                                                                                    label,
                                                                                    layers,
                                                                                    model_name)
    print("[1]", integrated_heatmaps1.shape)
    
    images2, integrated_heatmaps2 = heatmap_of_layer_before_last_integrand(device, inp,
                                                                           interpolation_on_activations_steps_arr,
                                                                           interpolation_on_images_steps_arr,
                                                                           label,
                                                                           layers[0] - 1,
                                                                           model_name)
    print("[2]", integrated_heatmaps2.shape)
    images3, integrated_heatmaps3 = heatmap_of_layer_before_last_integrand(device, inp,
                                                                           interpolation_on_activations_steps_arr,
                                                                           interpolation_on_images_steps_arr,
                                                                           label,
                                                                           layers[0] - 2,
                                                                           model_name)

    print("[3]", integrated_heatmaps3.shape)
    images, integrated_heatmaps1 = heatmap_of_layer_activations_integration(device, inp,
                                                                            interpolation_on_activations_steps_arr,
                                                                            interpolation_on_images_steps_arr,
                                                                            label,
                                                                            layers,
                                                                            model_name)

    print("[4]", integrated_heatmaps1.shape)
    integrated_heatmaps11 = F.interpolate(integrated_heatmaps1.unsqueeze(0).unsqueeze(0), size=(1024, 14, 14),
                                            mode='trilinear',
                                            align_corners=False).squeeze()
    integrated_heatmaps = integrated_heatmaps11
    heatmap = make_resize_norm(integrated_heatmaps)

    last_image = images[-1]

    t = tensor2cv(last_image.detach().cpu())
    shape = inp.shape[-2:]
    print("[5]", shape, t.shape)
    im, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(t, heatmap, use_mask=use_mask, H=shape[0], W=shape[1])

    return t, im, heatmap_cv, blended_img_mask, last_image, score, heatmap



def make_resize_norm(act_grads):
    heatmap = torch.sum(act_grads.squeeze(0), dim=0)
    heatmap = heatmap.unsqueeze(0).unsqueeze(0)

    heatmap = F.interpolate(heatmap, size=(224, 224), mode='bicubic', align_corners=False)
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    heatmap = heatmap.squeeze().cpu().data.numpy()
    return heatmap

def gen_dix(me, inp, catidx):

    start_time = time.time()
    if me.arch == "resnet50":
        FEATURE_LAYER_NUMBER = 7 ##8 ## 10000 ##8
    elif me.arch == "resnet50":
        FEATURE_LAYER_NUMBER = 8 ##8 ## 10000 ##8
    else:
        assert False, f"unexpected arch {me.arch}"
    INTERPOLATION_STEPS = 4
    USE_MASK = True
    
    interpolation_on_activations_steps_arr = [INTERPOLATION_STEPS]
    interpolation_on_images_steps_arr = [INTERPOLATION_STEPS]

    t, blended_im, heatmap_cv, blended_img_mask, image, score, heatmap = (
        get_by_class_saliency_dix(
            inp=inp,
            label=catidx,            
            model_name=me.arch,
            layers=[FEATURE_LAYER_NUMBER],
            interpolation_on_images_steps_arr=interpolation_on_images_steps_arr,
            interpolation_on_activations_steps_arr=interpolation_on_activations_steps_arr,
            device=inp.device,
            use_mask=USE_MASK))
    report_duration(start_time, me.arch, "DIX", '')
    return heatmap

class DixCnnSaliencyCreator:
    def __init__(self, alt_model=False):
        self.alt_model = alt_model

    def __call__(self, me, inp, catidx):           
        if self.alt_model:
            GradModel.model = me.model
        sal = gen_dix(me,inp,catidx)
        return {"DixCnn" : torch.tensor(sal).unsqueeze(0)}
    
