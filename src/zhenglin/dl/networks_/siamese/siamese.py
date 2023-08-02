from zhenglin.head import *
from zhenglin.dl.utils import fix_model
from zhenglin.dl.utils import EasyReplayBuffer
from zhenglin.dl.utils import LinearLambdaLR
from zhenglin.dl.networks.unet import UNet

from losses import ContrastiveLoss
from model import HalfUNet

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=400, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=192, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=192, help="high res. image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=5000, help="batch interval between model checkpoints")
parser.add_argument("--residual_blocks", type=int, default=9, help="number of residual blocks in the generator")
parser.add_argument("--warmup_batches", type=int, default=50, help="number of batches with pixel-wise loss only")
parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")

parser.add_argument('--dataroot', type=str, default='', help='')
parser.add_argument('--model_dir', type=str, default='', help='')
parser.add_argument('--version', type=str, default='v1_4_0_1', help='')
parser.add_argument("--step2save", type=int, default=10, help="Number of epoches to save a model")
parser.add_argument('--resume', action="store_true", help="train from latest checkpoints")
args = parser.parse_args()
print(str(args))

DEVICE = 0

model_save_path = os.path.join(args.model_dir, args.version)
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

shutil.copy(os.path.basename(__file__), 
            os.path.join(model_save_path, os.path.basename(__file__)))

patch_factor = 1
hr_shape = (args.hr_height // patch_factor, args.hr_width // patch_factor)

model = HalfUNet(1).to(DEVICE)

if args.resume is True:
    model.load_state_dict(torch.load('', map_location=torch.device(DEVICE)))
    print("model loaded.")

criterion_score = ContrastiveLoss(margin=1.0).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2))
lr_sheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=LinearLambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

train_set = DIYDataset(args.dataroot, patch_size=args.hr_height, scan_speed='any')
dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)

# ----------
#  Training
# ----------
for epoch in tqdm(range(args.epoch, args.n_epochs + 1)):
    same_cnt = 0
    diff_cnt = 0
    loss_in_epoch_same = 0
    loss_in_epoch_diff = 0
    loss_in_epoch = 0
    for i, imgs in enumerate(dataloader):
        batches_done = (epoch - args.epoch) * len(dataloader) + i
        img_A, img_B = imgs['A'], imgs['B']    # img_A and img_B could be the same or different
        label = 1 - imgs['loss']  # 1: same, 0: different, for contrastive loss
        
        img_A = Variable(img_A.type(Tensor)).to(DEVICE)
        img_B = Variable(img_B.type(Tensor)).to(DEVICE)
        label = Variable(label.type(Tensor)).to(DEVICE)
            
        # print('img_A', img_A.shape)
        # print('img_B', img_B.shape)
        
        # ------------------
        #  Train Generators
        # ------------------

        optimizer.zero_grad()
        
        f_A = model(img_A)
        f_B = model(img_B)
        
        distance = torch.sum(torch.pow(f_A - f_B, 2))
        
        loss = criterion_score(f_A, f_B, label)
        
        # plt.subplot(1, 2, 1)
        # plt.imshow(img_A[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.bone)
        # plt.subplot(1, 2, 2)
        # plt.imshow(img_B[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.bone)
        # s = 'same' if label.item() == 1 else 'different'
        # plt.suptitle(f'distance:{float(distance)}, loss:{loss.item()}, label:{int(label)}({s})')
        # plt.savefig(f"imgs/epoch_{epoch}_iter_{i}_img_AB.png", bbox_inches='tight')
        
        if int(label) == 0:
            diff_cnt += 1
            loss_in_epoch_diff += loss.item()
        if int(label) == 1:
            same_cnt += 1
            loss_in_epoch_same += loss.item()
        loss_in_epoch += loss.item()

        loss.backward()
        optimizer.step()

    
    lr_sheduler.step()
    
    print(f"{epoch}: {loss_in_epoch/(diff_cnt + same_cnt)}, same loss: {loss_in_epoch_same/same_cnt}, diff loss: {loss_in_epoch_diff/diff_cnt}")

    if epoch%(args.step2save) == 0 and epoch != 0:
        torch.save(model.state_dict(), os.path.join(args.model_dir, args.version + '/' + str(epoch) + '_model.pth'))