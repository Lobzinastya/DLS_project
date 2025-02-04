from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

def conv_simple(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3,
                 stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )

class MattingModule(nn.Module):
    def __init__(self, device, dim_inter_channel=64, lambda_l1=1, lambda_lap=1):
        super().__init__()
        self.device = device
        self.lambda_l1 = lambda_l1
        self.lambda_lap = lambda_lap

        self.layer2 = conv_simple(256 + 256, dim_inter_channel)  # все эти каналы и параметры из-за размерностей features, полученных от SAM2
        self.layer1 = conv_simple(256 + 64, dim_inter_channel)
        self.layer0 = conv_simple(256 + 32, dim_inter_channel)

        self.deconv2 = self._make_deconv(dim_inter_channel + 4, 1)
        self.deconv1 = self._make_deconv(dim_inter_channel + 4, 1)
        self.deconv0 = self._make_deconv(dim_inter_channel + 4, (dim_inter_channel + 4) // 2)
        self.deconv_final = self._make_deconv((dim_inter_channel + 4) // 2, 1, final=True)

        self._init_weights()
        self.to(device)

    def _make_deconv(self, in_channels, out_channels, final=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels,
                              kernel_size=3, stride=2,
                              padding=1, output_padding=1),
        ]
        if not final:
            layers += [
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @autocast()
    def forward(self, features, mask):

        mask = mask.to(self.device)
        images = features[0].to(self.device)
        backbone_fpn = [x.to(self.device) for x in features[1]['backbone_fpn']]
        vision_pos_enc = [x.to(self.device) for x in features[1]['vision_pos_enc']]

        combined2 = torch.cat((backbone_fpn[2], vision_pos_enc[2]), dim=1)
        x_os16 = self.layer2(combined2)
        mask_os16 = F.interpolate(mask, x_os16.shape[2:], mode='bilinear', align_corners=False)
        img_os16 = F.interpolate(images, x_os16.shape[2:], mode='bilinear', align_corners=False)
        x = self.deconv2(torch.cat((x_os16, img_os16, mask_os16), dim=1))
        os_8 = x

        combined1 = torch.cat((backbone_fpn[1], vision_pos_enc[1]), dim=1)
        x_os8 = self.layer1(combined1)
        img_os8 = F.interpolate(images, x_os8.shape[2:], mode='bilinear', align_corners=False)
        x = self.deconv1(torch.cat((x_os8, img_os8, os_8), dim=1))
        os_4 = x


        combined0 = torch.cat((backbone_fpn[0], vision_pos_enc[0]), dim=1)
        x_os4 = self.layer0(combined0)
        img_os4 = F.interpolate(images, x_os4.shape[2:], mode='bilinear', align_corners=False)
        x = self.deconv0(torch.cat((x_os4, img_os4, os_4), dim=1))
        os_1 = torch.sigmoid(self.deconv_final(x))


        os_4 = torch.sigmoid(os_4)
        os_8 = torch.sigmoid(os_8)
        return os_1, os_4, os_8

    def compute_loss(self, alpha_gt, alpha_os1, alpha_os4, alpha_os8):
        alpha_gt = alpha_gt.to(self.device)

        alpha_gt_os1 = F.interpolate(alpha_gt, alpha_os1.shape[2:], mode='bilinear', align_corners=False)
        alpha_gt_os4 = F.interpolate(alpha_gt, alpha_os4.shape[2:], mode='bilinear', align_corners=False)
        alpha_gt_os8 = F.interpolate(alpha_gt, alpha_os8.shape[2:], mode='bilinear', align_corners=False)

        l1_1 = F.l1_loss(alpha_os1, alpha_gt_os1)
        l1_4 = F.l1_loss(alpha_os4, alpha_gt_os4)
        l1_8 = F.l1_loss(alpha_os8, alpha_gt_os8)

        lap_1 = laplacian_loss(alpha_os1, alpha_gt_os1)
        lap_4 = laplacian_loss(alpha_os4, alpha_gt_os4)
        lap_8 = laplacian_loss(alpha_os8, alpha_gt_os8)

        l1_loss = l1_1 + l1_4 + l1_8
        lap_loss = lap_1 + lap_4 + lap_8

        # print(f"l1 {l1_1:.2f}, {l1_4:.2f}, {l1_8:.2f}, lap {lap_1:.2f}, {lap_4:.2f}, {lap_8:.2f}", end=' ')

        return self.lambda_l1 * l1_loss + self.lambda_lap * lap_loss

def laplacian_loss(pred, target):
    kernel = torch.tensor([[[[1, -2, 1],
                           [-2, 4, -2],
                           [1, -2, 1]]]],
                        dtype=torch.float32,
                        device=pred.device)
    kernel = kernel / kernel.abs().sum()

    pred_lap = F.conv2d(pred, kernel, padding=1)
    target_lap = F.conv2d(target, kernel, padding=1)

    return F.l1_loss(pred_lap, target_lap)



def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MattingModule(device, dim_inter_channel=16, lambda_l1=1, lambda_lap=1)
    save_path = 'app/utils/model_best_sad.pth'
    model.load_state_dict(torch.load(save_path))

    return model