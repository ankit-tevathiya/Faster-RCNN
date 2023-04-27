import torchvision
import torch
import numpy as np
from BoxHead import *
from utils import *
from pretrained_models import *
from train import *

if __name__ == '__main__':

    # Put the path were you save the given pretrained model
    pretrained_path='HW4_PartB_Code_Template/checkpoint680.pth'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    backbone, rpn = pretrained_models_680(pretrained_path,device=device)

    # we will need the ImageList from torchvision
    from torchvision.models.detection.image_list import ImageList

    # Put the path were the given hold_out_images.npz file is save and load the images
    hold_images_path='data/hold_out_images.npz'
    test_images=np.load(hold_images_path,allow_pickle=True)['input_images']


    # Put the path were you have your save network
    train_model_path='fasterrcnn-epochepoch=05-lossval_loss=0.32.ckpt'
    # Load your model here. If you use different parameters for the initialization you can change the following code
    # accordingly
    model = TrainModel().load_from_checkpoint(train_model_path)
    model.to(device)
    model.eval()

    keep_topK=200

    cpu_boxes = []
    cpu_scores = []
    cpu_labels = []

    for i, numpy_image in enumerate(test_images, 0):
        images = torch.from_numpy(numpy_image).to(device)
        backout = backbone(images)
        im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
        rpnout = rpn(im_lis, backout)
        proposals=[proposal[0:model.keep_topK,:] for proposal in rpnout[0]]
        fpn_feat_list= list(backout.values())
        feature_vectors = model.faster_rcnn.MultiScaleRoiAlign(fpn_feat_list,proposals)
        class_logits, box_pred = model.faster_rcnn.forward(feature_vectors)
        prediction = dict(labels=torch.zeros(len(class_logits)),class_logits=class_logits,regressor_target=box_pred,box_pred=box_pred,proposals=proposals)
        final_pred, final_conf, final_label = model.faster_rcnn.postprocess_detections(prediction, keep_num_postNMS=2)

        for box, score, label in zip(final_pred,final_conf,final_label):
            if box is None:
                cpu_boxes.append(None)
                cpu_scores.append(None)
                cpu_labels.append(None)
            else:
                cpu_boxes.append(box.to('cpu').detach().numpy())
                cpu_scores.append(score.to('cpu').detach().numpy())
                cpu_labels.append(label.to('cpu').detach().numpy())

    np.savez('predictions.npz', predictions={'boxes': cpu_boxes, 'scores': cpu_scores,'labels': cpu_labels})
