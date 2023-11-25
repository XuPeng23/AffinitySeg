import numpy as np
from utils.test_data import test_dataset
from utils.metric import cal_mae, cal_fm, cal_sm, cal_em, cal_wfm, cal_dice, cal_iou, cal_ber, cal_acc


def test_score(dataset_path, dataset_path_pre):
    sal_root = dataset_path_pre
    gt_root = dataset_path
    test_loader = test_dataset(sal_root, gt_root)
    mae, fm, sm, em, wfm, m_dice, m_iou, ber, acc = cal_mae(), cal_fm(
        test_loader.size), cal_sm(), cal_em(), cal_wfm(), cal_dice(), cal_iou(), cal_ber(), cal_acc()
    for i in range(test_loader.size):
        # print('predicting for %d / %d' % (i + 1, test_loader.size))
        sal, gt = test_loader.load_data()
        if sal.size != gt.size:
            x, y = gt.size
            sal = sal.resize((x, y))
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        gt[gt > 0.5] = 1
        gt[gt != 1] = 0
        res = sal
        res = np.array(res)
        if res.max() == res.min():
            res = res / 255
        else:
            res = (res - res.min()) / (res.max() - res.min())
        mae.update(res, gt)
        sm.update(res, gt)
        fm.update(res, gt)
        em.update(res, gt)
        wfm.update(res, gt)
        m_dice.update(res, gt)
        m_iou.update(res, gt)
        ber.update(res, gt)
        acc.update(res, gt)

    MAE = mae.show()
    maxf, meanf, _, _ = fm.show()
    sm = sm.show()
    em = em.show()
    wfm = wfm.show()
    m_dice = m_dice.show()
    m_iou = m_iou.show()
    ber = ber.show()
    acc = acc.show()
    print(
        'dataset: {} M_dice: {:.4f} M_iou: {:.4f} maxF: {:.4f} avgF: {:.4f} wfm: {:.4f} Sm: {:.4f} Em: {:.4f}  Ber: {:.4f}  Acc: {:.4f} MAE: {:.4f}'.format(
            'dataset:', m_dice, m_iou, maxf, meanf, wfm, sm, em, ber, acc, MAE))
    return m_iou, acc, m_dice


if __name__ == '__main__':
    dataset_path = r'W:\W_codes\medical_code\Segaware_ECA\data\TestDataset'
    dataset_path_pre = r'W:\W_codes\medical_code\Segaware_ECA\result_map\PolypPVT'
    test_score(['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'CVC-300'], dataset_path, dataset_path_pre)
