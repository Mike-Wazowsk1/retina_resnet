import argparse
import datetime

import cv2
import numpy as np
import torch

from ResNet18.resnet import ResNet18, make_emotion
from RetinaFace.retinaface import RetinaFace
from RetinaFace.utils import PriorBox, decode_landm, py_cpu_nms, decode
from cfg.cfg import Cfg
camera = cv2.VideoCapture(0)

def init_retina(cfg, path):
    if 'tar' in path:
        try:
            model = ResNet18()
            checkpoint = torch.load(f'{cfg.path}', map_location=torch.device(f'{cfg.device}'))
            model.load_state_dict(checkpoint['model_state_dict'])
            return model
        except Exception as e:
            print(f'There is something wrong: {e}')
    else:
        try:
            model = RetinaFace(cfg=cfg.retina_cfg)
            model.load_state_dict(torch.load(f'{path}', map_location=torch.device(f'{cfg.device}')))
            return model
        except Exception as e:
            print(f'There is something wrong: {e}')
    return None


def init_resnet(cfg,path):
    if 'tar' in path:
        try:
            model = ResNet18()
            checkpoint = torch.load(f'{path}', map_location=torch.device(f'{cfg.device}'))
            model.load_state_dict(checkpoint['model_state_dict'])
            return model
        except Exception as e:
            print(f'There is something wrong: {e}')
    else:
        try:
            model = ResNet18()
            model.load_state_dict(torch.load(f'{path}', map_location=torch.device(f'{cfg.device}')))
            return model
        except Exception as e:
            print(f'There is something wrong: {e}')
    return None


def convert_img(input_img):
    img = np.float32(input_img)
    im_height, im_width, _ = img.shape
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    return img


def detect_face(net=None, cfg=None, img_raw=None, debug=False):
    loc, conf, landms = net(convert_img(img_raw))
    device = torch.device('cpu')
    img = np.float32(img_raw)
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    im_height, im_width, _ = img.shape
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)

    scale = scale.to(device)
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    resize = 1
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > 0.4)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:750]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, 0.4)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:750, :]
    landms = landms[:750, :]
    dets = np.concatenate((dets, landms), axis=1)
    crop = None

    if True:
        for b in dets:
            if b[4] < 0.6:
                continue
            text = "{:.4f}".format(b[4])  # Model score
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)
            crop = img_raw[b[1]:b[3], b[0]:b[2]]

        if debug and type(crop) != type(None) and crop.shape[0] >= 100:
            return crop, (b[0], b[1]), (b[2], b[3])
        else:
            return crop, None, None


def start(name: str = None, group: str = None, debug=False, cfg=None) -> None:
    retina = init_retina(cfg=cfg, path=cfg.retina_path)
    resnet = init_resnet(cfg=cfg, path=cfg.resnet_path)
    if resnet is None or retina is None:
        print(f"""
        Error while loading model:
        Retina: {"Ok" if retina is not None else "Error"}
        Resnet: {"Ok" if resnet is not None else "Error"}""")


    while 1:
        ret, frame = camera.read()
        col = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
        if debug:
            face, rcx, rcy = detect_face(net=retina, img_raw=col, debug=debug,cfg=cfg.retina_cfg)
        else:
            face = detect_face(net=retina, img_raw=col, debug=debug,cfg=cfg.retina_cfg)[0]
        if type(face) != type(None) and face.shape[0] >= 100:
            crop = cv2.resize(face, (48, 48))
            crop_grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            x = np.array(crop_grey).reshape(1, 48, 48)
            emotion = make_emotion(resnet, x)

            cv2.imwrite(f'{cfg.path_to_img}/{"_".join(name)}_{"_".join(group)}_{emotion}_{datetime.datetime.now()}.jpeg', face)
            if debug:
                cv2.putText(frame, emotion, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)
                if rcx is None or rcy is None:
                    cv2.putText(frame, 'Face not recognizer', (256, 256), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA,
                                False)
                else:
                    if type(rcx[0] or rcx[1]) == np.float32 or type(rcy[0] or rcy[1]) == np.float32:
                        rcx = tuple(int(x) for x in rcx)
                        print('here')
                        rcy = tuple(int(y) for y in rcy)
                        cv2.rectangle(frame, rcx, rcy, (0, 0, 255), 1)
                    else:
                        print(type(rcx[0]),rcy)
                        print(rcx, rcy)
                        cv2.rectangle(frame, rcx, rcy, (0, 0, 255), 1)
                cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()


parser = argparse.ArgumentParser()
parser.add_argument("-n", '--name', required=False, default=None, type=str, help="name of employee", nargs='+')
parser.add_argument("-g", '--group', required=False, default=None, type=str, help="group of workers", nargs='+')
parser.add_argument('-d', '--debug', required=False, default=False, type=str,
                    help='Debug mode:y- Debug mode, n - default mode')
args = parser.parse_args()
name = args.name
group = args.group
cfg = Cfg()

if args.debug == 'y':
    debug = True
else:
    debug = False

if debug:
    start(name, group, debug, cfg)

