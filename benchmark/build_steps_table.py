from datetime import datetime

import torch
from tabulate import tabulate
from torch import nn
from torch.nn.utils import prune
from torchvision.models import SqueezeNet

import simplify
from simplify.utils import get_conv_bn, get_pinned_out
from tests import models
import pathlib
import re


def get_mark(passed):
    if isinstance(passed, bool):
        return ":heavy_check_mark:" if passed else ":x:"
    else:
        return ":cursing_face:" if str(
            passed) == "exception" else ":man_shrugging:"


if __name__ == '__main__':
    x = torch.randint(0, 256, (1, 3, 224, 224))
    x = x.float() / 255.
    input = torch.randint(0, 256, (256, 3, 224, 224))
    input = input.float() / 255.
    
    with torch.no_grad():
        for fuse in [True, False]:
            table = []
            for architecture in models:
                print(architecture.__name__)
                if architecture.__name__ in ["shufflenet_v2_x1_5", "shufflenet_v2_x2_0", "mnasnet0_75", "mnasnet1_3"]:
                    pretrained = False
                else:
                    pretrained = True
                if architecture.__name__ in ["inception_v3", "googlenet"]:
                    model = architecture(pretrained, transform_input=False, aux_logits=False)
                else:
                    model = architecture(pretrained)
                model.eval()
                
                for name, module in model.named_modules():
                    if isinstance(model, SqueezeNet) and 'classifier.1' in name:
                        continue
                    
                    if isinstance(module, nn.Conv2d):
                        prune.random_structured(
                            module, 'weight', amount=0.8, dim=0)
                        prune.remove(module, 'weight')
                
                y_src = model(input)
                
                print("BatchNorm Folding")
                try:
                    bn_folding = get_conv_bn(model)
                    if fuse:
                        model = simplify.fuse(model, bn_folding)
                    model.eval()
                    exception = None
                    y_dest = model(input)
                    passed_bf = torch.equal(y_src.argmax(dim=1), y_dest.argmax(dim=1))
                    print(
                        f'Correct predictions: {torch.eq(y_src.argmax(dim=1), y_dest.argmax(dim=1)).sum()}/{y_dest.shape[0]}')
                except Exception as e:
                    print("BatchNorm Folding")
                    print(architecture.__name__)
                    print(e)
                    passed_bf = "exception"
                
                if isinstance(passed_bf, bool) and passed_bf:
                    
                    pinned_out = get_pinned_out(model)
                    
                    print("Bias Propagation")
                    try:
                        simplify.propagate_bias(model, x, pinned_out)
                        model.eval()
                        exception = None
                        y_dest = model(input)
                        passed_bp = torch.equal(y_src.argmax(dim=1), y_dest.argmax(dim=1))
                        print(
                            f'Correct predictions: {torch.eq(y_src.argmax(dim=1), y_dest.argmax(dim=1)).sum()}/{y_dest.shape[0]}')
                    except Exception as e:
                        print("Bias Propagation")
                        print(architecture.__name__)
                        print(e)
                        passed_bp = "exception"
                    
                    if isinstance(passed_bp, bool) and passed_bp:
                        
                        print("Simplification")
                        try:
                            simplify.remove_zeroed(model, x, pinned_out)
                            model.eval()
                            exception = None
                            y_dest = model(input)
                            passed_simp = torch.equal(y_src.argmax(dim=1), y_dest.argmax(dim=1))
                            print(
                                f'Correct predictions: {torch.eq(y_src.argmax(dim=1), y_dest.argmax(dim=1)).sum()}/{y_dest.shape[0]}')
                        except Exception as e:
                            print("Simplification")
                            print(architecture.__name__)
                            print(e)
                            passed_simp = "exception"
                    else:
                        passed_simp = "skipped"
                else:
                    passed_bp, passed_simp = "skipped", "skipped"
                
                table.append([architecture.__name__, get_mark(passed_bf), get_mark(passed_bp), get_mark(passed_simp)])
            table = tabulate(table,
                             headers=['Architecture', 'BatchNorm Folding', 'Bias Propagation', 'Simplification'],
                             tablefmt='github', stralign="center")
            
            root = pathlib.Path(__file__).parent.parent.resolve()
            
            if fuse:
                index_re = re.compile(r"<!\-\- table fuse starts \-\->.*<!\-\- table fuse ends \-\->", re.DOTALL)
            else:
                index_re = re.compile(r"<!\-\- table no fuse starts \-\->.*<!\-\- table no fuse ends \-\->", re.DOTALL)
            
            updated = "Update timestamp " + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\n"
            
            if fuse:
                index = ["<!-- table fuse starts -->", updated, table, "<!-- table fuse ends -->"]
            else:
                index = ["<!-- table no fuse starts -->", updated, table, "<!-- table no fuse ends -->"]
                
            readme = root / "README.md"
            index_txt = "\n".join(index).strip()
            readme_contents = readme.open().read()
            readme.open("w").write(index_re.sub(index_txt, readme_contents))
