### Easy follow procedure can be found in FasterRCNN.ipynb uploaded (follow step by step)

# DataProcessing 
Use Dataset.py - to call the DataSet creation and dataLoaders 

# BoxHead
All BoxHead related functions are found in BoxHead.py 
Included ROIAlign, training modules, Postporcessing for a single image or batch of images

# Train 
Contains lightening module code for Training and plotting - this class instantiates the BoxHead class and trains the model

# Utils 
Contains extra functions required for the processing and plotting - iou calculation, output decoding using proposals, mAP/AP calculations


## Steps:
1. Do the needed imports

    from HW4_PartB_Code_Template.pretrained_models import *
    from HW4_PartB_Code_Template.BoxHead import *
    from HW4_PartB_Code_Template.utils import *
    from HW4_PartB_Code_Template.train import *


2. call the DataSet creation and dataLoaders

    dataset = BuildDataset(paths)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    batch_size = 5
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

3. Instantiate the Model - using TrainModel() (which instantiates the BoxHead within it)

    model = TrainModel()
    model.to(device)

4. Train model  (simply call trainer on the lightening module)

    trainer = pl.Trainer(gpus=0, max_epochs=15)
    trainer.fit(model, train_loader, test_loader)

    Plot losses using - model.loss_plots() 

5. Use model and use different functions to plot and visualise the proposals 
    
    #### to plot the proposals 
    with torch.no_grad():
        batch = next(iter(test_loader))
        images, labels, _, bboxes, _  = batch.values() 
        prediction = model(batch)

        conf_score,pred_labels = torch.max(F.softmax(prediction['class_logits'],dim=1),dim=1)

        indices = torch.where(prediction['labels']>0)
        print("Total Class Accuracy\t\t: ",np.round((torch.where(pred_labels==prediction['labels'],1,0).sum()/len(pred_labels)*100).item(),2))
        print("No Background - Class Recall\t: ",np.round((torch.where(prediction['labels'][indices] == pred_labels[indices],1,0).sum()/len(indices[0])).item()*100,2))
        indices = torch.where(pred_labels>0)
        print("No Background - Class Precision\t: ",np.round((torch.where(prediction['labels'][indices] == pred_labels[indices],1,0).sum()/len(indices[0])).item()*100,2))

        pos_indices = torch.where(pred_labels>0)[0]
        all_indices = (pred_labels[pos_indices].reshape(-1,1)-1)*4
        all_indices = torch.hstack((all_indices,all_indices+1,all_indices+2,all_indices+3))
        box_pred = prediction['box_pred'][pos_indices]
        box_pred = output_decoding(torch.stack([box_pred[i][all_indices[i]] for i in range(len(pos_indices))]),torch.vstack(prediction['proposals'])[pos_indices], device=device)
        gt_box_pred = output_decoding(prediction['regressor_target'][pos_indices],torch.vstack(prediction['proposals'])[pos_indices], device=device)
        
        keep_topK = 200
        image_index = torch.div(pos_indices, keep_topK, rounding_mode='trunc')
        plot_anchors(images,box_pred,gt_box_pred,pred_labels[pos_indices].cpu().detach().numpy(),image_index)
        final_pred, final_conf, final_label = model.faster_rcnn.postprocess_detections(prediction,keep_num_preNMS=50, keep_num_postNMS=3)
        plot_predictions(images,final_pred,final_label,bboxes)


    #### to get final predicitons 
    with torch.no_grad():
        images, labels, _, bboxes, _  = batch.values() 
        prediction = model(batch)
        final_pred, final_conf, final_label = model.faster_rcnn.postprocess_detections(prediction,keep_num_preNMS=50, keep_num_postNMS=3)