from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.autograd import Variable
from sklearn.model_selection import KFold
import cv2

from test_score import test_score
from image_loader_TNBC512 import *
from Network import *


cudnn.benchmark = True
torch.cuda.set_device(0)
bce = nn.BCELoss()
bce_logits = nn.BCEWithLogitsLoss().cuda()


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def single_gpu_train(batch_size=4, dataset='TNBC'):

    # path
    method = 'Ours'
    model_path = './model/{}/'.format(dataset)
    validation_pred_path = './temp/validation_results/{}'.format(dataset)
    validation_gt_path = './temp/validation_gt/{}'.format(dataset)
    test_pred_path = './temp/test_results/{}'.format(dataset)
    test_gt_path = './temp/test_gt/{}'.format(dataset)

    training_root = './data/{}/train/'.format(dataset)
    test_root = './data/{}/test/'.format(dataset)

    # Model
    net = Model().cuda().train()

    # Train Image List
    train_img_name_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(training_root, 'images'))
                           if f.endswith('.jpg') or f.endswith('.png')]
    train_image_list = [(os.path.join(training_root, 'images', img_name + '.png'), os.path.join(training_root, 'masks',
                                                                                                img_name + '.png')) for
                        img_name in train_img_name_list]
    # Test Image List
    test_img_name_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(test_root, 'images'))
                          if f.endswith('.jpg') or f.endswith('.png')]
    test_image_list = [(os.path.join(test_root, 'images', img_name + '.png'),
                        os.path.join(test_root, 'masks', img_name + '.png')) for img_name in test_img_name_list]
    test_set = ImageFolder(test_image_list, mode='test',
                           joint_transform=val_joint_transform,
                           img_transform=img_transform,
                           label_transform=label_transform)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False)
    print('Test Data Loaded')

    # *********************** Cross_validation_iteration *****************************
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    idx = [w for w in range(len(train_image_list))]  # Index list for splitting the dataset

    cross_validation_iteration = 0
    for train_idx, validation_idx in kfold.split(idx):
        cross_validation_iteration = cross_validation_iteration + 1
        print('cross_validation_iteration:{}'.format(cross_validation_iteration))
        print('train_idx len:{}'.format(len(train_idx)))
        print('validation_idx len:{}'.format(len(validation_idx)))

        net = Model().cuda().train()

        optimizer = torch.optim.Adamax([{'params': net.parameters()}], lr=5e-4)

        train_list_fold = []
        validation_list_fold = []

        for index in train_idx.tolist():
            train_list_fold.append(train_image_list[index])
        for index in validation_idx.tolist():
            validation_list_fold.append(train_image_list[index])

        train_set = ImageFolder(train_list_fold, mode='train',
                                joint_transform=joint_transform,
                                img_transform=img_transform,
                                label_transform=label_transform)

        validation_set = ImageFolder(validation_list_fold, mode='test',
                                     joint_transform=val_joint_transform,
                                     img_transform=img_transform,
                                     label_transform=label_transform)

        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)
        print('Train Data Loaded')
        validation_loader = DataLoader(validation_set, batch_size=1, num_workers=0, shuffle=True, drop_last=True)
        print('Validation Data Loaded')

        max_mdice = 0

        epoch_loss_txt = ''
        epoch_dice_txt = ''
        epoch_miou_txt = ''

        for epoch in range(300):
            epoch = epoch
            net.train()
            print('Start epoch[{}/300]'.format(epoch))
            epoch_loss = 0

            # ************************* Train ***************************
            for i, train_data in enumerate(train_loader):
                inputs, labels, _ = train_data
                labels = labels[:, 0, :, :].unsqueeze(1)
                inputs = Variable(inputs).cuda()
                labels = Variable(labels).cuda()
                optimizer.zero_grad()

                dist_loss, out = net(inputs, labels)
                loss = bce(out, labels) + dist_loss

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(str(epoch) + ' epoch_loss', epoch_loss)
            epoch_loss_txt = epoch_loss_txt + '{:.4f},'.format(epoch_loss)

            # save model parameters for each epoch
            with open("log/epoch_loss.txt", "w") as f:
                f.write(epoch_loss_txt)

            # ********************** Validation **************************
            net.train(False)
            net.eval()

            # Get one Test miniBatch
            for i, test_data in enumerate(validation_loader):
                inputs, label, fnames = test_data
                inputs = Variable(inputs).cuda()

                _, pred = net(inputs)
                pred = pred.squeeze(1).detach().cpu().numpy()

                # Save prediction of one minibatch
                for j in range(0, len(fnames)):
                    _, cur_fname = os.path.split(fnames[j])
                    cur_fname, _ = os.path.splitext(cur_fname)
                    cur_fname = cur_fname + '.png'
                    gt = label[j][0].squeeze(1).detach().cpu().numpy()
                    gt[gt > 0.5] = 255
                    gt[gt <= 0.5] = 0

                    out = cv2.resize(pred[j], dsize=gt.shape[1::-1])
                    out[out > 0.5] = 255
                    out[out <= 0.5] = 0
                    # Save predict results
                    cv2.imwrite(validation_pred_path + '/{}/{}'.format(cross_validation_iteration, cur_fname), out)
                    cv2.imwrite(validation_gt_path + '/{}/{}'.format(cross_validation_iteration, cur_fname), gt)

            miou, acc, mdice = test_score(validation_pred_path + '/{}'.format(cross_validation_iteration),
                                          validation_gt_path + '/{}'.format(cross_validation_iteration))

            # dice
            epoch_dice_txt = epoch_dice_txt + '{:.4f},'.format(mdice)
            with open("log/epoch_dice.txt", "w") as f:
                f.write(epoch_dice_txt)
            # miou
            epoch_miou_txt = epoch_miou_txt + '{:.4f},'.format(miou)
            with open("log/epoch_miou.txt", "w") as f:
                f.write(epoch_miou_txt)

            if mdice > max_mdice:
                max_mdice = mdice
                print('saving best_{}.PTH'.format(epoch))
                saved_model_name = 'iteration_{}_best_{}_'.format(cross_validation_iteration, dataset) + method + '.PTH'
                torch.save(net.state_dict(), model_path + saved_model_name)
                with open(model_path + saved_model_name.split('.')[0] + '_epoch_{}_mdice_{:.4f}_miou_{:.4f}'.format(
                        epoch, mdice, miou) + ".txt", "w") as f:
                    f.write('_')
    print('Training End')

    # Test ##################################################

    net.train(False)
    net.eval()

    sumDice = 0
    sumIOU = 0
    test_mDice = ''
    test_mIOU = ''

    for it in range(5):
        net.load_state_dict(torch.load('model/{}/iteration_{}_best_{}_{}.PTH'.format(dataset, it+1, dataset, method)))
        print('loading test model...')

        # Get one Test miniBatch
        for i, test_data in enumerate(test_loader):
            inputs, _, fnames = test_data
            inputs = Variable(inputs).cuda()

            _, pred = net(inputs)
            pred = pred.squeeze(1).detach().cpu().numpy()

            # Save prediction of one minibatch
            for j in range(0, len(fnames)):
                _, cur_fname = os.path.split(fnames[j])
                cur_fname, _ = os.path.splitext(cur_fname)
                cur_fname = cur_fname + '.png'
                gt = cv2.imread(os.path.join(test_root, 'masks/' + cur_fname))
                out = cv2.resize(pred[j], dsize=gt.shape[1::-1])
                out[out > 0.5] = 255
                out[out <= 0.5] = 0
                # Save predict results
                cv2.imwrite(test_pred_path + '/{}/'.format(it+1) + cur_fname, out)
                cv2.imwrite(test_gt_path + '/{}/'.format(it+1) + cur_fname, gt)
        # Evaluation
        miou, acc, mdice = test_score(test_pred_path + '/{}'.format(it+1),
                                      test_gt_path + '/{}'.format(it+1))
        test_mDice = test_mDice + '_{}_{}'.format(it+1, mdice)
        test_mIOU = test_mIOU + '_{}_{}'.format(it+1, miou)
        sumDice = sumDice + mdice
        sumIOU = sumIOU + miou

    # Save log
    with open(model_path + "test_result.txt", "w") as f:
        f.write(test_mDice)
        f.write(test_mIOU)
        f.write('    mDice:{}'.format(sumDice/5))
        f.write('    mIOU:{}'.format(sumIOU/5))


if __name__ == '__main__':

    # _dataset = 'Glas'
    _dataset = 'TNBC'

    single_gpu_train(batch_size=4, dataset=_dataset)
