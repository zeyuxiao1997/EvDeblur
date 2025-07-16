import torch
import torch.nn as nn
import math
from torch.nn import functional as F
import basicsr.models.archs.arch_util as arch_util
# import arch_util as arch_util
from einops import rearrange

# 

class DuIntNet(nn.Module):
    def __init__(self):
        super(DuIntNet, self).__init__()

        ks = 3 # kernel size   
        self.window_size = 7*4
        # self.nf = arg.nf
        ########## multi-scale feature extractor for RGB & event ##########
        self.conv_rgb_1_1 = arch_util.conv(3, 32, kernel_size=ks, stride=1)
        self.conv_rgb_1_2 = nn.Sequential(arch_util.resnet_block(32, kernel_size=ks),
                                          arch_util.ResASPP(32),
                                          arch_util.resnet_block(32, kernel_size=ks),
                                          arch_util.ResASPP(32),
                                          )
        self.conv_rgb_1_3 = nn.Sequential(arch_util.resnet_block(32, kernel_size=ks),
                                          arch_util.ResASPP(32),
                                          arch_util.resnet_block(32, kernel_size=ks),
                                          arch_util.ResASPP(32),
                                          )

        self.conv_rgb_2_1 = arch_util.conv(32, 64, kernel_size=ks, stride=2)
        self.conv_rgb_2_2 = nn.Sequential(arch_util.resnet_block(64, kernel_size=ks),
                                          arch_util.ResASPP(64),
                                          arch_util.resnet_block(64, kernel_size=ks),
                                          arch_util.ResASPP(64),
                                          )
        self.conv_rgb_2_3 = nn.Sequential(arch_util.resnet_block(64, kernel_size=ks),
                                          arch_util.ResASPP(64),
                                          arch_util.resnet_block(64, kernel_size=ks),
                                          arch_util.ResASPP(64),
                                          )

        self.conv_rgb_3_1 = arch_util.conv(64, 128, kernel_size=ks, stride=2)
        self.conv_rgb_3_2 = nn.Sequential(arch_util.resnet_block(128, kernel_size=ks),
                                          arch_util.ResASPP(128),
                                          arch_util.resnet_block(128, kernel_size=ks),
                                          arch_util.ResASPP(128),
                                          )
        self.conv_rgb_3_3 = nn.Sequential(arch_util.resnet_block(128, kernel_size=ks),
                                          arch_util.ResASPP(128),
                                          arch_util.resnet_block(128, kernel_size=ks),
                                          arch_util.ResASPP(128),
                                          )

        self.conv_event_1_1 = arch_util.conv(6, 32, kernel_size=ks, stride=1)
        self.conv_event_1_reduce = arch_util.conv(32, 16, kernel_size=ks, stride=1)
        self.conv_event_1_expand = arch_util.conv(16, 32, kernel_size=ks, stride=1)
        self.conv_event_1_2 = arch_util.conv(32, 32, kernel_size=ks, stride=1)
        self.conv_event_1_3 = arch_util.conv(32, 32, kernel_size=ks, stride=1)
        self.conv_event_1_4 = arch_util.conv(32, 32, kernel_size=ks, stride=1)

        self.conv_event_2_1 = arch_util.conv(32, 64, kernel_size=ks, stride=2)
        self.conv_event_2_reduce = arch_util.conv(64, 32, kernel_size=ks, stride=1)
        self.conv_event_2_expand = arch_util.conv(32, 64, kernel_size=ks, stride=1)
        self.conv_event_2_2 = arch_util.conv(64, 64, kernel_size=ks, stride=1)
        self.conv_event_2_3 = arch_util.conv(64, 64, kernel_size=ks, stride=1)
        self.conv_event_2_4 = arch_util.conv(64, 64, kernel_size=ks, stride=1)

        self.conv_event_3_1 = arch_util.conv(64, 128, kernel_size=ks, stride=2)
        self.conv_event_3_reduce = arch_util.conv(128, 64, kernel_size=ks, stride=1)
        self.conv_event_3_expand = arch_util.conv(64, 128, kernel_size=ks, stride=1)
        self.conv_event_3_2 = arch_util.conv(128, 128, kernel_size=ks, stride=1)
        self.conv_event_3_3 = arch_util.conv(128, 128, kernel_size=ks, stride=1)
        self.conv_event_3_4 = arch_util.conv(128, 128, kernel_size=ks, stride=1)

        # ######################## EventModulation #############################
        self.attention3_1 = arch_util.attention(dim=128)
        self.attention3_2 = arch_util.attention(dim=128)
        self.attention3_3 = arch_util.attention(dim=128)
        self.fusion3 = arch_util.fusion(dim=128)

        self.attention2_1 = arch_util.attention(dim=64)
        self.attention2_2 = arch_util.attention(dim=64)
        self.fusion2 = arch_util.fusion(dim=64)

        self.attention1_1 = arch_util.attention(dim=32)
        self.attention1_2 = arch_util.attention(dim=32)
        self.fusion1 = arch_util.fusion(dim=32)
        
       
        ############################# Decoder #############################
        self.upconv3_i = arch_util.conv(128, 128, kernel_size=ks, stride=1)
        self.upconv3_2 = nn.Sequential(arch_util.resnet_block(128, kernel_size=ks),
                                          arch_util.ResASPP(128),
                                          arch_util.resnet_block(128, kernel_size=ks),
                                          arch_util.ResASPP(128),
                                          )
        self.upconv3_1 = nn.Sequential(arch_util.resnet_block(128, kernel_size=ks),
                                          arch_util.ResASPP(128),
                                          )

        self.upconv2_u = arch_util.upconv(128, 64)
        self.upconv2_i = arch_util.conv(128, 64, kernel_size=ks,stride=1)
        self.upconv2_2 = nn.Sequential(arch_util.resnet_block(64, kernel_size=ks),
                                          arch_util.ResASPP(64),
                                          arch_util.resnet_block(64, kernel_size=ks),
                                          arch_util.ResASPP(64),
                                          )
        self.upconv2_1 = nn.Sequential(arch_util.resnet_block(64, kernel_size=ks),
                                          arch_util.ResASPP(64),
                                          )

        self.upconv1_u = arch_util.upconv(64, 32)
        self.upconv1_i = arch_util.conv(64, 32, kernel_size=ks,stride=1)
        self.upconv1_2 = nn.Sequential(arch_util.resnet_block(32, kernel_size=ks),
                                          arch_util.ResASPP(32),
                                          arch_util.resnet_block(32, kernel_size=ks),
                                          arch_util.ResASPP(32),
                                          arch_util.resnet_block(32, kernel_size=ks),
                                          arch_util.ResASPP(32),
                                          arch_util.resnet_block(32, kernel_size=ks),
                                          arch_util.ResASPP(32),
                                          arch_util.resnet_block(32, kernel_size=ks),
                                          arch_util.ResASPP(32),
                                          )
        self.upconv1_1 = nn.Sequential(arch_util.resnet_block(32, kernel_size=ks),
                                          arch_util.ResASPP(32),
                                          arch_util.resnet_block(32, kernel_size=ks),
                                          arch_util.ResASPP(32),
                                          arch_util.resnet_block(32, kernel_size=ks),
                                          arch_util.ResASPP(32),
                                          arch_util.resnet_block(32, kernel_size=ks),
                                          arch_util.ResASPP(32),
                                          arch_util.resnet_block(32, kernel_size=ks),
                                          arch_util.ResASPP(32),
                                          arch_util.resnet_block(32, kernel_size=ks),
                                          arch_util.ResASPP(32),
                                          arch_util.resnet_block(32, kernel_size=ks),
                                          arch_util.ResASPP(32),
                                          arch_util.resnet_block(32, kernel_size=ks),
                                          arch_util.ResASPP(32),
                                          arch_util.resnet_block(32, kernel_size=ks),
                                          arch_util.ResASPP(32),
                                          arch_util.resnet_block(32, kernel_size=ks),
                                          arch_util.ResASPP(32),
                                          )

        self.img_prd = arch_util.conv(32, 3, kernel_size=ks, stride=1)


    def forward(self, frame, event):
        _,_,h,w = frame.shape
        frame = self.check_image_size(frame)
        event = self.check_image_size(event)
        # encoder
        # sharing weights
        frame_L2 = F.interpolate(frame, scale_factor=0.5, mode="bilinear")
        frame_L3 = F.interpolate(frame_L2, scale_factor=0.5, mode="bilinear")

        conv1_frame = self.conv_rgb_1_3(self.conv_rgb_1_2(self.conv_rgb_1_1(frame)))
        conv2_frame = self.conv_rgb_2_3(self.conv_rgb_2_2(self.conv_rgb_2_1(conv1_frame)))
        conv3_frame = self.conv_rgb_3_3(self.conv_rgb_3_2(self.conv_rgb_3_1(conv2_frame)))
        
        conv1_event = self.conv_event_1_3(self.conv_event_1_2(self.conv_event_1_1(event)))
        conv2_event = self.conv_event_2_3(self.conv_event_2_2(self.conv_event_2_1(conv1_event)))
        conv3_event = self.conv_event_3_3(self.conv_event_3_2(self.conv_event_3_1(conv2_event)))

        conv3_frame = rearrange(conv3_frame, 'b c h w -> b h w c')
        conv3_event = rearrange(self.conv_event_3_reduce(conv3_event), 'b c h w -> b h w c')

        out_frame3, out_event3 = self.attention3_1(conv3_frame, conv3_event)
        out_frame3, out_event3 = self.attention3_2(out_frame3, out_event3)
        out_frame3, out_event3 = self.attention3_3(out_frame3, out_event3)

        out_frame3 = rearrange(out_frame3, 'b h w c -> b c h w')
        out_event3 = rearrange(out_event3, 'b h w c -> b c h w')
        out_event3 = self.conv_event_3_4(self.conv_event_3_expand(out_event3))

        out_level3 = self.fusion3(out_frame3, out_event3) + out_frame3


        conv2_frame = rearrange(conv2_frame, 'b c h w -> b h w c')
        conv2_event = rearrange(self.conv_event_2_reduce(conv2_event), 'b c h w -> b h w c')
        
        out_frame2, out_event2 = self.attention2_1(conv2_frame, conv2_event)
        out_frame2, out_event2 = self.attention2_2(out_frame2, out_event2)

        out_frame2 = rearrange(out_frame2, 'b h w c -> b c h w')
        out_event2 = rearrange(out_event2, 'b h w c -> b c h w')
        out_event2 = self.conv_event_2_4(self.conv_event_2_expand(out_event2))

        out_level2 = self.fusion2(out_frame2, out_event2) + out_frame2


        conv1_frame = rearrange(conv1_frame, 'b c h w -> b h w c')
        conv1_event = rearrange(self.conv_event_1_reduce(conv1_event), 'b c h w -> b h w c')

        out_frame1, out_event1 = self.attention1_1(conv1_frame, conv1_event)
        out_frame1, out_event1 = self.attention1_2(out_frame1, out_event1)


        out_frame1 = rearrange(out_frame1, 'b h w c -> b c h w')
        out_event1 = rearrange(out_event1, 'b h w c -> b c h w')
        out_event1 = self.conv_event_1_4(self.conv_event_1_expand(out_event1))

        out_level1 = self.fusion1(out_frame1, out_event1) + out_frame1


        cat3 = self.upconv3_i(out_level3) # torch.Size([2, 64, 128, 128])
        upconv2 = self.upconv2_u(self.upconv3_1(self.upconv3_2(cat3)))  # torch.Size([2, 64, 128, 128])
        # print(upconv2.shape,out_level2.shape)
        cat2 = self.upconv2_i(torch.cat((upconv2,out_level2),1))   # torch.Size([2, 64, 128, 128])
        upconv1 = self.upconv1_u(self.upconv2_1(self.upconv2_2(cat2)))  # torch.Size([2, 32, 256, 256])
        cat1 = self.upconv1_i(torch.cat((upconv1,out_level1),1))   # torch.Size([2, 32, 256, 256])
        img_prd = self.img_prd(self.upconv1_1(self.upconv1_2(cat1)))+frame    # torch.Size([2, 3, 256, 256])

        return img_prd[:,:,:h,:w]
    

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x




