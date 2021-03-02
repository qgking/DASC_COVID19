from module.Attention_block import *
from module.Res_block import *



class RAUnet(nn.Module):
    def __init__(self, n_channels, n_classes, filter_num=8):
        super(RAUnet, self).__init__()

        self.atu_conv1 = nn.Sequential(
            nn.Conv3d(n_channels, filter_num * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(filter_num * 4),
            nn.ReLU(inplace=True)
        )
        self.covmpool = nn.MaxPool3d(kernel_size=2)
        self.res1 = residual_block(input_channels=filter_num * 4, output_channels=filter_num * 4)
        self.mpool1 = nn.MaxPool3d(kernel_size=2)
        self.res2 = residual_block(input_channels=filter_num * 4, output_channels=filter_num * 8)
        self.mpool2 = nn.MaxPool3d(kernel_size=2)
        self.res3 = residual_block(input_channels=filter_num * 8, output_channels=filter_num * 16)
        self.mpool3 = nn.MaxPool3d(kernel_size=2)
        self.res4 = residual_block(input_channels=filter_num * 16, output_channels=filter_num * 32)
        self.mpool4 = nn.MaxPool3d(kernel_size=2)
        self.res5 = nn.Sequential(
            residual_block(input_channels=filter_num * 32, output_channels=filter_num * 64),
            residual_block(input_channels=filter_num * 64, output_channels=filter_num * 64)
        )
        self.atb1 = attention_block(input_channels=filter_num * 32, output_channels=filter_num * 32, encoder_depth=1)
        self.res6 = residual_block(input_channels=(filter_num * (64 + 32)), output_channels=filter_num * 32)
        self.atb2 = attention_block(input_channels=filter_num * 16, output_channels=filter_num * 16,
                                    encoder_depth=2)
        self.res7 = residual_block(input_channels=(filter_num * (32 + 16)), output_channels=filter_num * 16)
        self.atb3 = attention_block(input_channels=filter_num * 8, output_channels=filter_num * 8, encoder_depth=3)
        self.res8 = residual_block(input_channels=(filter_num * (16 + 8)), output_channels=filter_num * 8)
        self.atb4 = attention_block(input_channels=filter_num * 4, output_channels=filter_num * 4, encoder_depth=4)
        self.res9 = residual_block(input_channels=(filter_num * (8 + 4)), output_channels=filter_num * 4)
        self.atu_conv2 = nn.Sequential(
            nn.Conv3d(filter_num * (4 + 4), filter_num * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(filter_num * 4),
            nn.ReLU(inplace=True)
        )
        # self.atu_conv3 = nn.Sequential(
        #     nn.Conv3d(filter_num * 4, 1, kernel_size=3, padding=1),
        #     nn.Tanh()
        # )
        self.atu_conv3 = nn.Conv3d(filter_num * 4, n_classes, kernel_size=3, padding=1)

        # self.init_weights()
        self.apply(weights_init_normal)

    def forward(self, x):
        atu_conv1 = self.atu_conv1(x)
        covmpool = self.covmpool(atu_conv1)
        res1 = self.res1(covmpool)
        mpool1 = self.mpool1(res1)
        res2 = self.res2(mpool1)
        mpool2 = self.mpool2(res2)
        res3 = self.res3(mpool2)
        mpool3 = self.mpool3(res3)
        res4 = self.res4(mpool3)
        mpool4 = self.mpool4(res4)
        res5 = self.res5(mpool4)
        up1 = F.interpolate(res5, scale_factor=2, mode="trilinear", align_corners=True)
        atb1 = self.atb1(res4)
        merge1 = torch.cat((up1, atb1), dim=1)
        res6 = self.res6(merge1)
        atb2 = self.atb2(res3)
        up2 = F.interpolate(res6, scale_factor=2, mode="trilinear", align_corners=True)
        merge2 = torch.cat((up2, atb2), dim=1)
        res7 = self.res7(merge2)
        atb3 = self.atb3(res2)
        up3 = F.interpolate(res7, scale_factor=2, mode="trilinear", align_corners=True)
        merge3 = torch.cat((up3, atb3), dim=1)
        res8 = self.res8(merge3)
        atb4 = self.atb4(res1)
        up4 = F.interpolate(res8, scale_factor=2, mode="trilinear", align_corners=True)
        merge4 = torch.cat((up4, atb4), dim=1)
        res9 = self.res9(merge4)
        up5 = F.interpolate(res9, scale_factor=2, mode="trilinear", align_corners=True)
        merge5 = torch.cat((up5, atu_conv1), dim=1)
        atu_conv2 = self.atu_conv2(merge5)
        atu_conv3 = self.atu_conv3(atu_conv2)
        # return atu_conv3
        return torch.sigmoid(atu_conv3)
