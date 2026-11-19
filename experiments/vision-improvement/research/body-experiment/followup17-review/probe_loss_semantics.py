"""CPU-only executable assertions for the installed 17-slot loss semantics."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'runtime-config'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'matplotlib'))
import torch
import ultralytics.utils.loss as loss_module
from ultralytics.utils.loss import KeypointLoss
from ultralytics.utils.metrics import OKS_SIGMA

torch.set_num_threads(2)
sigmas = torch.tensor(OKS_SIGMA, dtype=torch.float64, device='cpu')
pred = torch.ones((1,17,3),dtype=torch.float64,requires_grad=True,device='cpu')
truth = torch.zeros((1,17,3),dtype=torch.float64,device='cpu')
truth[:,5:,2] = 2
mask = truth[...,2] != 0
area = torch.tensor([[10000.]],dtype=torch.float64,device='cpu')
criterion = KeypointLoss(sigmas)
coordinate = criterion(pred,truth,mask,area)
altered = truth.clone(); altered[:,:5,:2] = 1000
assert torch.equal(coordinate,criterion(pred,altered,mask,area))
coordinate.backward()
assert torch.count_nonzero(pred.grad[:,:5,:2]) == 0
assert torch.count_nonzero(pred.grad[:,5:,:2]) == 24
# The same body sigmas give the same average coordinate loss with 12 slots.
body_only = KeypointLoss(sigmas[5:])(pred.detach()[:,5:],truth[:,5:],mask[:,5:],area)
# Mask-count normalization is computed at float32 precision internally.
assert torch.allclose(coordinate,body_only,rtol=1e-6,atol=1e-9)
visibility_logits = torch.zeros((1,17),dtype=torch.float64,requires_grad=True,device='cpu')
visibility = torch.nn.functional.binary_cross_entropy_with_logits(visibility_logits,mask.to(torch.float64))
visibility.backward()
assert torch.all(visibility_logits.grad[:,:5] > 0)
assert torch.all(visibility_logits.grad[:,5:] < 0)
report = {'device':'cpu','status':'passed','faceCoordinatesAreExcluded':True,
          'faceCoordinateGradientNonzeros':0,'bodyCoordinateGradientNonzeros':24,
          'masked17CoordinateLoss':coordinate.item(),'matchedSigma12CoordinateLoss':body_only.item(),
          'coordinateAverageAbsoluteDifference':abs(coordinate.item()-body_only.item()),
          'faceVisibilityIsNegativeSupervision':True,
          'bodyVisibilityIsPositiveSupervision':True,
          'COCO17Sigmas':sigmas.tolist(),
          'note':'Retaining COCO sigmas differs from the first 12-slot recipe uniform1/12; this is a recipe-bundle comparison.',
          'installedLossSource':str(Path(loss_module.__file__).resolve()),
          'installedLossSHA256':hashlib.sha256(Path(loss_module.__file__).read_bytes()).hexdigest(),
          'probeSHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(ROOT/'loss-semantics.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
