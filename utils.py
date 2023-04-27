import numpy as np
import torch
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.patches as patches

device='cuda:0'

def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))

# This function computes the IOU between two set of boxes
def single_box_iou(boxA, boxB):
    '''
    compute the IOU between the boxA, boxB boxes
    '''
    x1, y1 = torch.max(boxA[0], boxB[0]), torch.min(boxA[1], boxB[1])
    x2, y2 = torch.min(boxA[2], boxB[2]), torch.max(boxA[3], boxB[3])
    intersection = torch.abs(x1-x2) * torch.abs(y1-y2)  
    ar1 = torch.abs(boxA[0]-boxA[2]) * torch.abs(boxA[1]-boxA[3])
    ar2 = torch.abs(boxB[0]-boxB[2]) * torch.abs(boxB[1]-boxB[3])
    return intersection/(ar1+ar2-intersection)

def iou(bbox, possible_anchor):
    '''
    Vectorized IoU for all pred, target
    '''
    pred_area = ((bbox[:,3]-bbox[:,1]) * (bbox[:,2]-bbox[:,0]).T).reshape(-1,1)
    area_gt = ((possible_anchor[:,3]-possible_anchor[:,1]) * (possible_anchor[:,2]-possible_anchor[:,0]).T).reshape(-1,1)
    intersect_width = torch.maximum(torch.minimum(bbox[:,2].reshape(-1,1),possible_anchor[:,2].reshape(-1,1).T)- torch.maximum(bbox[:,0].reshape(-1,1),possible_anchor[:,0].reshape(-1,1).T),torch.tensor(0))
    intersect_height = torch.maximum(torch.minimum(bbox[:,3].reshape(-1,1),possible_anchor[:,3].reshape(-1,1).T)- torch.maximum(bbox[:,1].reshape(-1,1),possible_anchor[:,1].reshape(-1,1).T),torch.tensor(0))
    # compute intersect area
    area_intersect = intersect_width * intersect_height
    # compute union area
    area_union = (pred_area + area_gt.T) - area_intersect
    return (area_intersect / area_union)

def output_decoding(regressor_target,proposals, device='cuda:0'):
    '''
    This function decodes the output of the box head that are given in the [t_x,t_y,t_w,t_h] format
    into box coordinates where it return the upper left and lower right corner of the bbox
    Input:
        regressed_boxes_t: (total_proposals,4) ([t_x,t_y,t_w,t_h] format)
        flatten_proposals: (total_proposals,4) ([x1,y1,x2,y2] format)
    Output:
        box: (total_proposals,4) ([x1,y1,x2,y2] format)
    '''
    if type(proposals)==list:
        proposals = torch.vstack(proposals).clone()
    w = torch.exp(regressor_target[:,2]) * (proposals[:,2] - proposals[:,0])
    h = torch.exp(regressor_target[:,3]) * (proposals[:,3] - proposals[:,1])
    x = regressor_target[:,0]*(proposals[:,2] - proposals[:,0]) + (proposals[:,0]+proposals[:,2])/2
    y = regressor_target[:,1]*(proposals[:,3] - proposals[:,1]) + (proposals[:,1]+proposals[:,3])/2
    box = torch.stack((x-(w/2), y-(h/2), x+(w/2), y+(h/2)),dim=1)
    return box.to(device)


def plot_anchors(image,proposals,decoded,labels,image_bounding_boxes_map):
    '''Plot selected anchors'''
    # class Vehicle - red, class People - blue, class Animals - green  
    c = ['purple','blue','green'] 
    c_a = ['mediumorchid','cornflowerblue','springgreen'] 
    thickness = 3

    for i in range(image.shape[0]):
        img = (image[i].permute(1,2,0).numpy()).copy()
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        bb_ind = torch.where(image_bounding_boxes_map==i)[0]
        for bc in bb_ind:
            bc = bc.item()
            c_index = labels[bc]-1
            x1,y1,x2,y2 = decoded[bc]
            rect=patches.Rectangle((int(x1),int(y1)),int(x2-x1),int(y2-y1), 
                                    fill=False,
                                    color=c[c_index],
                                    linewidth=thickness)
            plt.gca().add_patch(rect)
            rx, ry = rect.get_xy()
            plt.annotate(labels[bc], (rx, ry), color='w', weight='bold',fontsize=10, ha='center', va='center')
    

            x1,y1,x2,y2 = proposals[bc]
            rect=patches.Rectangle((int(x1),int(y1)),int(x2-x1),int(y2-y1), 
                                fill=False,
                                color=c_a[c_index],
                                linewidth=thickness)
            plt.gca().add_patch(rect)
        plt.title("True Bounding Boxes and Proposals")
        plt.show()

    
def plot_predictions(image,final_pred,final_label,bboxes):
    '''Plot selected anchors'''
    # class Vehicle - red, class People - blue, class Animals - green  
    c = ['purple','blue','green'] 
    c_a = ['mediumorchid','cornflowerblue','springgreen'] 
    thickness = 3

    for i in range(image.shape[0]):
        img = (image[i].permute(1,2,0).numpy()).copy()
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        
        for j in range(len(final_pred[i])):
            c_index = final_label[i][j]-1
            x1,y1,x2,y2 = final_pred[i][j]
            rect=patches.Rectangle((int(x1),int(y1)),int(x2-x1),int(y2-y1), 
                                    fill=False,
                                    color=c[c_index],
                                    linewidth=thickness)
            plt.gca().add_patch(rect)
            rx, ry = rect.get_xy()
            # ann = str(final_label[i][j].item())+" :"+str(np.round(final_conf[i][j].item(),2))
            plt.annotate("p", (rx, ry), color='w', weight='bold',fontsize=6, ha='center', va='center')
        
        for j in range(len(bboxes[i])):
            x1,y1,x2,y2 = bboxes[i][j]
            rect=patches.Rectangle((int(x1),int(y1)),int(x2-x1),int(y2-y1), 
                                fill=False,
                                color=c_a[c_index],
                                linewidth=thickness)
            plt.gca().add_patch(rect)
        
        plt.title(f"Prediction Label: {np.unique(final_label[i].detach().numpy())[0]}")
        plt.show()

def iou_to_TP(iou,score,th=0.5):
  '''IOU to TP matrix'''
  if(iou.shape[1]<1):
    return [],0
  ind = torch.argsort(score, descending=True) 
  tp = torch.max(torch.where(iou>th,1,0)[ind],axis=1)[0]
  arg = torch.argmax(torch.where(iou>th,1,0)[ind],axis=1)
  return tp, arg,iou.size()[1]

def pr_curve(scores,tp_list,arg_list,gts):
  '''
  Precision recall curves
  '''
  score = torch.tensor(scores,device=device)
  tps = torch.tensor(tp_list,device=device)
  args = torch.torch.tensor(arg_list, device=device)

  rec, current_rec, prec, current_prec = [],[],[],[]

  for ctr,x in enumerate(tps):
    ind = torch.where(x == 1.)[0].tolist()
    
    if ind:
      current_prec.append(ind)

      if args[ctr] not in current_rec:
        current_rec.append(args[ctr])
    
    rec.append(len(current_rec) / gts[ctr])
    prec.append(len(current_prec) / (ctr+1))

  return torch.tensor(prec,device=device),torch.tensor(rec,device=device)

def average_precision(prec_rec):
  ind = torch.argsort(prec_rec[1], descending=True)

  d = torch.abs(torch.diff(prec_rec[1][ind]))
  pr = torch.cummax(prec_rec[0][ind],dim=0)[0][1:]
  return torch.sum(d * pr)

def mean_average_precision(ap_class_1,ap_class_2,ap_class_3):
  return (ap_class_1+ap_class_2+ap_class_3)/3

def results(labels,pred):
  '''Given Labels and predicitons get all results - PR curve, AP and mAP'''

  calculate = [iou(labels[i],pred[i]) for i in range(len(labels))]

  tp_list1, tp_list2, tp_list3 = [], [], []
  arg_list1, arg_list2, arg_list3 = [], [], []
  gts1, gts2, gts3 = [0], [0], [0]
  score1, score2, score3 = [], [], []

  for i in range(len(calculate)):
    # class 0
    temp = iou_to_TP(calculate[i][0], calculate[i][1])
    if(len(temp[0])>0):
      tp_list1.extend(temp[0])
      arg_list1.extend(temp[1])
      gts1.extend([temp[2]+gts1[-1]]*temp[0].shape[0])
      score1.extend(calculate[i][1])
    # class 1
    temp = iou_to_TP(calculate[i][2], calculate[i][3])
    if(len(temp[0])>0):
      tp_list2.extend(temp[0])
      arg_list2.extend(temp[1])
      gts2.extend([temp[2]+gts2[-1]]*temp[0].shape[0])
      score2.extend(calculate[i][3])
    # class 2
    temp = iou_to_TP(calculate[i][4], calculate[i][5])
    if(len(temp[0])>0):
      tp_list3.extend(temp[0])
      arg_list3.extend(temp[1])
      gts3.extend([temp[2]+gts3[-1]]*temp[0].shape[0])
      score3.extend(calculate[i][5])
  
  gts1.pop(0)
  gts2.pop(0)
  gts3.pop(0)

  pr_class_1 = pr_curve(score1,tp_list1, arg_list1 , gts1)
  pr_class_2 = pr_curve(score2,tp_list2, arg_list2 , gts2)
  pr_class_3 = pr_curve(score3,tp_list3, arg_list3 , gts3)

  ap_class_1 = average_precision(pr_class_1).item()
  ap_class_2 = average_precision(pr_class_2).item()
  ap_class_3 = average_precision(pr_class_3).item()

  mean_ap = (ap_class_1+ap_class_2+ap_class_3)/3

  return {'ap':[ap_class_1,ap_class_2,ap_class_3],
          'pr': [pr_class_1,pr_class_2,pr_class_3],
          'mean_ap':mean_ap
          }
