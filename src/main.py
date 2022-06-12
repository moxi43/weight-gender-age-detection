# imports
import numpy as np
import torch
import cv2
from torchvision import transforms
from network_weight import UNet
from network import UNet as HUNet
import argparse
from draw_skeleton import create_colors, draw_skeleton

if __name__ == "__main__":
    # PARSER SETTINGS
    np.random.seed(23)
    parser = argparse.ArgumentParser(description="Height and Weight Information from Unconstrained Images")

    parser.add_argument('-i', '--image', type=str, required=True, help='Image Directory')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU selection')
    parser.add_argument('-r', '--resolution', type=int, required=True, help='Resolution for Square Image')
    args = parser.parse_args()

    model_w = UNet(128, 32, 32)
    pretrained_model_w = torch.load('models/model_ep_37.pth.tar', map_location=torch.device('cpu'))
    
    model_w.load_state_dict(pretrained_model_w["state_dict"]) 
    
    if torch.cuda.is_available():
        model = model_w.cuda(args.gpu)
    else:
        model = model_w
    
    # Reading Image 
    assert ".jpg" in args.image or ".png" in args.image or ".jpeg" in args.image, "Please use .jpg or .png format"
    
    RES = args.resolution
    
    X = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB).astype('float32')
    scale = RES / max(X.shape[:2])
    
    X_scaled = cv2.resize(X, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) 
    
    if X_scaled.shape[1] > X_scaled.shape[0]:
        p_a = (RES - X_scaled.shape[0])//2
        p_b = (RES - X_scaled.shape[0])-p_a
        X = np.pad(X_scaled, [(p_a, p_b), (0, 0), (0,0)], mode='constant')
    elif X_scaled.shape[1] <= X_scaled.shape[0]:
        p_a = (RES - X_scaled.shape[1])//2
        p_b = (RES - X_scaled.shape[1])-p_a
        X = np.pad(X_scaled, [(0, 0), (p_a, p_b), (0,0)], mode='constant') 
    
    o_img = X.copy()
    X /= 255
    X = transforms.ToTensor()(X).unsqueeze(0)
        
    if torch.cuda.is_available():
        X = X.cuda()
    
    model.eval()
    with torch.no_grad():
        m_p, j_p, _, w_p = model(X)
    
    del model

    fformat = '.png'

    if '.jpg' in args.image:
        fformat = '.jpg'
    elif '.jpeg' in args.image:
        fformat = '.jpeg'   

    mask_out = m_p.argmax(1).squeeze().cpu().numpy()
    joint_out = j_p.argmax(1).squeeze().cpu().numpy()
    pred_2 = j_p.squeeze().cpu().numpy()

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_out.astype('uint8'))
    colors = create_colors(30)
    img_sk = np.zeros((128,128,3))
    
    joint_pos = []
        
    for i in range(1, num_labels):
        p_res = np.expand_dims((labels==i).astype(int),0) * pred_2
        
        ct_ = 1
        positions = []

        for i in range(1,19):
            positions.append(np.unravel_index(p_res[ct_,:,:].argmax(), p_res[ct_,:,:].shape))
            ct_ += 1
            
        joint_pos.append(positions)
    
    mask_out_RGB = np.concatenate([255*mask_out[:, :, np.newaxis],
                                  255*mask_out[:, :, np.newaxis],
                                  mask_out[:, :, np.newaxis],
                                  ], axis=-1)
    
    layer = cv2.addWeighted(o_img.astype('uint8'), 0.55, mask_out_RGB.astype('uint8'), 0.45, 0) 
    img_sk = draw_skeleton(layer/255, joint_pos, colors)

    out_name = args.image.split("/")[-1].replace(fformat, '.mask.png')
    out_name_j = args.image.split("/")[-1].replace(fformat, '.joint.png')
    out_name_sk = args.image.split("/")[-1].replace(fformat, '.skeleton.png')

    
    print("\nImage: " + args.image)
    print("Weight: {:.1f} kg".format(100*w_p.item()))

    #gc.collect()
    ## AGE AND GENDER PART

    from deepface import DeepFace

    obj = DeepFace.analyze(img_path = args.image, actions = ['age', 'gender'])
    print("\nGender: " + obj["gender"])
    print("\nAge: " + str(obj["age"]))

   