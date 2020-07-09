
#########################################################################################################################
##
##                                          Person Detection script 
##
#########################################################################################################################

# Initating the libraries 
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys

## Queue class

class Queue:
    '''
    Class for dealing with queues requests
    
    Performs basic operations for queues like adding to a queue, getting the queues 
    and checking the coordinates for queues.
    
    Labels - add_queue, get_queues, check_coords
    
    the queues are used to the generates the coordinates of the tracking object and perform frames operations. 
    
    '''
    def __init__(self):
        '''
        Initalizing the queue algorithm
        '''
        self.queues=[]

    def add_queue(self, points):
        '''
        Input: Points 
        
        Output: list (Points)
        Adding the points data in list
        '''
        self.queues.append(points)

    def get_queues(self, image):
        '''
        Input: Image 
        
        Output: frames 
        
        The queues generated from images are passed to the yield of frames
        '''
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield (frame)
    
    def check_coords(self, coords):
        '''
        input: coords
        
        Output: d 
        
        the coords find the range queues in the data and check the coordinates of the input data. 
        '''
        d={k+1:0 for k in range(len(self.queues))}
        
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0]>q[0] and coord[2]<q[2]:
                    d[i+1]+=1
        return (d)

## Person Detection Class 
    
class PersonDetect:
    '''
    Class for the Person Detection Model.
    
    Program for data preprocessing and detecting of person using Intel Open Model Zoo. 
    
    Input_Attributes:
        
        model_weights: A model weights path in bin format.
        model_structure: A model structure path in xml format.
        device: A device which perform task on the processor {CPU, GPU, Myraid, FPGA} .
        IEcore: coreCore represents an Inference Engine.
        model: Load model object for Intermediate Representation.
        input_name: A input list for the image or the video.
        input_shape: the input shape is the size or resolution the tuple input.
        output_shape: the output shape is the generated pre-process coordinate in tuple shape.
        output_name: the output names is the out source list of image or frames data.
        threshold: A threshold value is set the floting limits of the .
        
        --model ${Load model} 
        --device ${Select the device type} 
        --video ${Input video file (mp4, mpeg, MKV)} 
        --queue_param ${setting queue parameter for limit} 
        --output_path ${Set-up output path directory}
        --max_people ${Limiting maximum people} 

    '''
    
    def __init__(self, model_name, device, threshold=0.60):
        '''
        Initalizing the code for the people detection in queue. 
        '''
        # Initialize the model input parameters
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
        
        
        ## Initialize frame class variables
        # Setting the values to zeros 
        self.w = (0.0) #frame width 
        self.h = (0.0) #frame height 
        self.input_frame = None # set input frame to None 
        self.case_frame = None # set case frame to None for later use
        self.model= None # set model for None 
        self.ex_model= None #set executable model network to None
        
        # Initialize the inference engine for the running the code. (issue - model was not loading using IENetwork)
        # [Solution](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html#afe73d64ddd115a41f5acc0d31031f52b)
        self.core= IECore() # Inference Engine Plugin
        
        try:
            self.model= self.core.read_network(self.model_structure, self.model_weights) # use read_network instead of IEnetwork
            #self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
            
        # Getting the input layer for the model
        '''
        [doc](https://docs.openvinotoolkit.org/2018_R5/_ie_bridges_python_docs_api_overview.html)
        '''
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        
        #self.output_shape=self(model.output[self_name])
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        TODO: This method needs to be completed by you
        
        Load the inference model in function. 
        
        [doc](https://docs.openvinotoolkit.org/latest/classInferenceEngine_1_1Core.html)
        
        load network: 
                    network = model file 
                    device_name = device type 
                    num_request = 1 set default
        
        '''
        # Load the model network
        self.ex_model= self.core.load_network(network=self.model,device_name=self.device,num_requests=1)
        
        #self.core = IECore()
        #self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
    
    
    def async_req_get(self, input_image):
        '''
        Start an asynchronous request
        It is about running the primary application thread seperate 
        
        input: input_image 
        
        Output: result directory 
        '''
        
        self.ex_model.start_async(request_id=0,inputs=input_image) # asynchrnous model start
        
        #the request get hold untill is gets complete operation 
        
        status = self.ex_model.requests[0].wait(-1) # keeping the status for the model request
        
        if status==0:
            # Extract and return the output results
            result = self.ex_model.requests[0].outputs[self.output_name] # outpu_name request throgh model
            
            return (result)
        
   
    def predict(self, image):
        '''
        TODO: This method needs to be completed by you
        
        Prediction alogorithm 
        the program use for pre-processing input image and getting the coordinate data and output image. 
        
        Input: image 
        
        Output: coords and output_image 
        '''
        
        self.input_frame = image
        
        #the input frame is passed to the prediction 
        
        self.case_frame = self.preprocess_input(self.input_frame)
        input_image={self.input_name: self.case_frame}
        
        #the preprocessing of the image data is extracted
        
        result = self.async_req_get(input_image) # result of the prediction
        coords, output_image = self.preprocess_outputs(result) # coordinates of the person detected
        
        
        return (coords,output_image)
    
    def draw_outputs(self, coords, image):
        '''
        TODO: This method needs to be completed by you
        
        The draw output algoritm generate the bounding boxes in the image with respect to the coordinate image. 
        
        [doc](https://docs.opencv.org/3.4/da/d0c/tutorial_bounding_rects_circles.html)
        
        input: coords and image 
        
        output: image outcome using opencv. 
        '''
        
        pass1 = (coords[0],coords[1])
        pass2 = (coords[2],coords[3])
        
        # Drawing bounding boxes on the image input
        
        cv2.rectangle(image, pass1, pass2, (255, 0, 0) , 2)
        
        return

    def preprocess_outputs(self, outputs):
        '''
        TODO: This method needs to be completed by you
        
        Pre-processing is the image processing techniques to applying the image processing techniques and adding box output
        for reference. The threshold is set at enhances image features at detected person. 
        
        input: outputs
        
        output: coordinates, input_frame
        '''
        
        coordinates=list()
        
        #In the box in represent as {outputs[0][0]}:
        
        for b in range (len(outputs[0][0])):
            box = outputs[0][0][b]
            confidence = box[2]
            if confidence > self.threshold:
                x_min,x_max = map(lambda b : int(b*self.w), [box[3],box[5]])
                y_min,y_max = map(lambda b : int(b*self.h), [box[4],box[6]])
                coordinates.append([x_min,y_min,x_max,y_max])
                coords = [x_min,y_min,x_max,y_max]
                self.draw_outputs(coords, self.input_frame)
                
        return (coordinates, self.input_frame)

    def preprocess_input(self, image):
        '''
        TODO: This method needs to be completed by you
        
        Pre-processing input takes image and pass the feature operation for providing the input parameter shape to model. 
        this is used to resize the input frame. 
        
        input: image 
        
        output: pass_image
        '''
        
        # Pre-process of the input image 
        
        _width=self.input_shape[3]
        _height=self.input_shape[2]
        
        pass_image = cv2.resize(image, (_width, _height))
        pass_image = pass_image.transpose((2,0,1))
        pass_image = pass_image.reshape(self.input_shape[0], self.input_shape[1], _height, _width)
        
        return (pass_image)

## Main file processing script

def main(args):
    
    model=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path

    start_model_load_time=time.time()
    pd = PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue=Queue()
    
    try:
        queue_param=np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("error loading queue param file")

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
        
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pd.w = initial_w
    pd.h = initial_h
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h))
    
    counter=0
    start_inference_time=time.time()

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1
            
            coords, image= pd.predict(frame)
            num_people= queue.check_coords(coords)
            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
            out_text=""
            y_pixel=25
            
            for k, v in num_people.items():
                out_text += f"No. of People in Queue {k} is {v} "
                if v >= int(max_people):
                    out_text += f" Queue full; Please move to next Queue "
                cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 60), 2)
                out_text=""
                y_pixel+=40
            out_video.write(image)
            
        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')

        out_video.release()
        cap.release()
        cv2.destroyAllWindows()
    
    except Exception as e:
        
        print (fps)
        print (total_inference_time)
        print("Could not run Inference: ", e)

## Main script for execution 
        
if __name__=='__main__':
    
    parser=argparse.ArgumentParser()
    
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--device', default='CPU', type=str)
    parser.add_argument('--video', default=None, type=str)
    parser.add_argument('--queue_param', default=None, type=str)
    parser.add_argument('--output_path', default='/results', type=str)
    parser.add_argument('--max_people', default=2, type=int)
    parser.add_argument('--threshold', default=0.60, type=float)
    
    args=parser.parse_args()

    
    main(args)