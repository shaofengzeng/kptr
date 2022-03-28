#-*-coding:utf8-*-
import torch
import torch.nn as nn
import kornia
from einops.einops import rearrange
from solver.nms import box_nms
from model.modules.cnn.vgg_backbone import VGGBackbone,VGGBackboneBN
from model.modules.cnn.cnn_heads import DetectorHead, DescriptorHead
from model.modules.transformer.transformer import LocalFeatureTransformer

class KPTR(torch.nn.Module):
    """ Pytorch definition of kptr """

    def __init__(self, config, input_channel=1, grid_size=8, device='cpu', using_bn=True):
        super(KPTR, self).__init__()
        self.nms = config['nms']
        self.det_thresh = config['det_thresh']
        self.topk = config['topk']
        self.device = device
        if using_bn:
            self.backbone = VGGBackboneBN(config['backbone']['vgg'], input_channel, device=device)
        else:
            self.backbone = VGGBackbone(config['backbone']['vgg'], input_channel, device=device)
        ##

        self.att = LocalFeatureTransformer(config['transformer'])

        self.detector_head = DetectorHead(input_channel=config['det_head']['feat_in_dim'],
                                          grid_size=grid_size,
                                          using_bn=using_bn)
        self.descriptor_head = DescriptorHead(input_channel=config['des_head']['feat_in_dim'],
                                              output_channel=config['des_head']['feat_out_dim'],
                                              grid_size=grid_size,
                                              using_bn=using_bn)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        if isinstance(x, dict):
            feat_map = self.backbone(x['img'])
        else:
            feat_map = self.backbone(x)
        det_outputs = self.detector_head(feat_map)

        prob = det_outputs['prob']
        if self.nms is not None:
            prob = [box_nms(p.unsqueeze(dim=0),
                            self.nms,
                            min_prob=self.det_thresh,
                            keep_top_k=self.topk).squeeze(dim=0) for p in prob]
            prob = torch.stack(prob)
            det_outputs.setdefault('prob_nms',prob)

        pred = prob[prob>=self.det_thresh]
        det_outputs.setdefault('pred', pred)

        ##descriptor
        fb,fc,fh,fw = feat_map.shape
        #tf_feat = feat_map.contiguous().view(fb, fc, -1)
        tf_feat = rearrange(feat_map, 'n c h w -> n (h w) c')

        if isinstance(x, dict):
            tf_mask = x['mask'] if 'mask' in x else torch.ones(fb, fh, fw, device=self.device)
        else:
            tf_mask = torch.ones(fb, fh, fw, device=self.device)
        tf_mask = kornia.resize(tf_mask, (fh, fw), align_corners=True)
        tf_mask = tf_mask.flatten(-2)
        tf_feat = self.att(tf_feat, tf_mask)
        tf_feat = rearrange(tf_feat, 'n (h w) c -> n c h w', h=fh, w=fw)
        desc_input_feat = tf_feat.contiguous().view(fb, fc, fh, fw)

        desc_outputs = self.descriptor_head(desc_input_feat)
        return {'det_info':det_outputs, 'desc_info':desc_outputs}
