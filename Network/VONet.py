import torch 
import torch.nn as nn
import torch.nn.functional as F

class VONet(nn.Module):
    def __init__(self, fix_parts=('flow', 'stereo')):
        super(VONet, self).__init__()

        from .PWC import PWCDCNet as FlowNet
        self.flowNet = FlowNet(uncertainty=False)

        from .StereoNet7 import StereoNet7 as StereoNet
        self.stereoNet = StereoNet()

        from .VOFlowNet import VOFlowRes as FlowPoseNet
        self.flowPoseNet = FlowPoseNet(intrinsic=True, down_scale=True, stereo=0, fix_parts=fix_parts)
        # from .orig_VOFlowNet import VOFlowRes as FlowPoseNet
        # self.flowPoseNet = FlowPoseNet(intrinsic=True, down_scale=True, config=1, stereo=0)
        # only optimize the pose network
        if "flow" in fix_parts:
            for param in self.flowNet.parameters():
                param.requires_grad = False
            
            for param in self.GIM.parameters():
                param.requires_grad = False

        if "stereo" in fix_parts:
            for param in self.stereoNet.parameters():
                param.requires_grad = False

    def update_GIM(self):
        from .dkm.models.model_zoo.DKMv3 import DKMv3
        self.GIM = DKMv3(weights=None, h=672, w=896)

    def forward(self, img0, img1, img0_norm, img0_r_norm, intrinsic, flowNet_model='GIM'):
        # import ipdb;ipdb.set_trace()
        with torch.no_grad():   # not optimize the flow and stereo network
            if flowNet_model == 'PWC':
                flow, _ = self.flowNet(torch.cat([img0, img1], dim=1))
                flow = flow[0]  # [8, 2, H/4, W/4]
            elif flowNet_model == 'GIM':
                flow = []
                for i in range(img0.shape[0]):
                    query = img0[i].unsqueeze(0)
                    support = img1[i].unsqueeze(0)
                    batch = {"query": query, "support": support}
                    flow_ = self.GIM.forward_symmetric(batch, batched = True)
                    flow_ = flow_[4]['dense_flow'][0:1]
                    flow.append(flow_)
                flow = torch.cat(flow, dim=0)

            # predict disparity (inverse depth)
            disp, _ = self.stereoNet(torch.cat((img0_norm, img0_r_norm),dim=1))
            disp = F.interpolate(disp, scale_factor=0.25, mode='nearest')
        
        x = torch.cat([flow, intrinsic], dim=1)
        pose = self.flowPoseNet(x)

        return flow, disp, pose
