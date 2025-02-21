import os
import torch
import torch.nn.utils.prune as prune
from mmpose.apis import init_model

# https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
# https://pytorch.org/docs/stable/nn.html

config = '/mmlab/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192.py'
checkpoint = '/mmlab/mmpose/models/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth'

model = init_model(config, checkpoint, device='cpu') # 'cuda'

def count_parameters(model):
    counter = 0
    for name, param in model.named_parameters():
        counter += 1
        print(f'{name}: {type(param)}, {param.size()}')
    return counter

print(f'Total parameters before pruning: {count_parameters(model)}')

is_depth_wise = True

if is_depth_wise:
    for name, module in model.named_modules():
        print(f'{name}\n')

    # Manuel removing entire blocks
    # Just an extreme example. 
    # Normally you wouldn't want to remove an entire 
    # backbone stage due to loss of functionality ...
    model.backbone.stage4 = torch.nn.Identity()

    # Remove the top 2 layers
    # layers = list(model.children())
    # new_layers = layers[2:]
    # model = torch.nn.Sequential(*new_layers)

    for name, module in model.named_modules():
        print(f'{name}\n')
else:
    # width wise
    def apply_pruning(model, amount=0.5):
        for name, module in model.named_modules():
            # print(f'{name}\n')                
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):   
                # Prune tensor by removing channels with the lowest Ln-norm along the specified dimension.         
                # prune.ln_structured(module, name='weight', amount=amount, n=2, dim=2)

                # Prune tensor by removing units with the lowest L1-norm
                prune.l1_unstructured(module, name='weight', amount=amount)

                # Prune tensor by removing random (currently unpruned) units.
                # prune.random_unstructured(module, name='weight', amount=amount)

                # Make the pruning permanent
                prune.remove(module, "weight")

    apply_pruning(model, 0.3)

print(f'Total parameters after pruning: {count_parameters(model)}')

def get_model_size(model, filename='pruned_model.pth'):
    torch.save(model.state_dict(), filename)
    size = os.path.getsize(filename) / (1024 * 1024)
    return size

original_size = os.path.getsize(checkpoint) / (1024 * 1024)
pruned_size = get_model_size(model)

print(f'original_size: {original_size:.2f} MB')
print(f'pruned_size: {pruned_size:.2f} MB')
