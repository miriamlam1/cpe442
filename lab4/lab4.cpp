#include "opencv2/videoio.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h> 
#include <iostream>
#include <opencv2/imgproc.hpp>
 
using namespace cv;
using namespace std;
 
// init functions
void init_frames(VideoCapture vc);
void to442_greyscale(Mat frame);
void to442_sobel(Mat in);
void* grey_and_sobel(Mat frame);
Mat divide_image(Mat frame);
 
#define WAIT 10
#define NUM_THREADS 4
#define BUFFER 2 // sobel overlap in pixels. should be 2
 
static Mat kernelx = (Mat_<char>(3, 3) << -1, 0, 1,
                   -2, 0, 2,
                    - 1, 0, 1);
static Mat kernely = (Mat_<char>(3, 3) << 1, 2, 1,
                   0, 0, 0,
                   -1, -2, -1);
 
static int y = 0;
int height;
 
// main
int main(int argc, char **argv){
    if (argc < 2){
        cout << "empty path" << endl;
        return -1; // fail
    }
    VideoCapture vc(argv[1]);
    init_frames(vc);
    return 0; // success
}
 
void init_frames(VideoCapture vc){
    Mat frame;
    String windowName = "Multi-Threaded Sobel";
    vc >> frame;
    height = frame.rows;
    
    while (!frame.empty()){
     
        // Do stuff with frame
        frame = divide_image(frame);
 
        // Output frame
        imshow(windowName, frame);
 
        // waits WAIT ms and checks if escapekey is pressed to escape
        if (waitKey(WAIT) == 27)
            break;
            
        vc >> frame;
    }
}
 
// each quarter runs this thread
void* grey_and_sobel(void* frame_ptr){
    
    Mat frame = *(Mat*)(frame_ptr);
    to442_greyscale(frame);
    to442_sobel(frame);
    
    pthread_exit(NULL); // exit because its done with thread at this point
}
 
Mat divide_image(Mat frame){
    int i; // counter
    Mat quarter;
    Mat quarter_arr[NUM_THREADS]; // list of the 4 quarters
    Mat quarter_arr2[NUM_THREADS]; // output quarters
    pthread_t threads[NUM_THREADS]; // list of threads
 
    // left to right: splitting like this [1|2|3|4]
    int newcols = frame.cols/NUM_THREADS;
    
    // first frame only have right overlap
    Mat copy = frame.clone(); // copy to not sobel overlap original frame
    quarter_arr[0] = copy(Rect(0,0,newcols+BUFFER,height));
    
    // this for loop separates into quarters
    for(i=1; i<NUM_THREADS; i++){
        int x = newcols*i - BUFFER;
        int width;
        
        // last frame only have left overlap
        if(i == NUM_THREADS-1){
            width = newcols+BUFFER;
            copy = frame.clone();
            quarter_arr[i] = copy(Rect(x,y,newcols+BUFFER,height));
        // middle frames overlap left and right
        } else {
            width = newcols + BUFFER*2;
            copy = frame.clone();
            quarter_arr[i] = copy(Rect(x,y,width,height));
        }
    }
    
     // do the threads
    for(i=0; i<NUM_THREADS; i++){
        pthread_create (&threads[i], NULL, grey_and_sobel, &(quarter_arr[i]));
    }
 
    // crop the image to output
    quarter_arr2[0] = quarter_arr[0](Rect(0,0,newcols,height));
    for(int i=1;i<NUM_THREADS;i++){
         quarter_arr2[i] = quarter_arr[i](Rect(BUFFER,0,newcols,height));
    }
 
    // wait for all 4 threads to terminate
    for(int i=0;i<NUM_THREADS;i++){
        if(pthread_join(threads[i],NULL)<0)
            cout << "Thread join error" << endl;
    }
        
    // concatenate the images together
    Mat out;
    // hconcat(quarter_arr2, out); THIS COMMAND DOESNT WORK WHY
    hconcat(quarter_arr2[0],quarter_arr2[1], out);
    for(i=2;i<NUM_THREADS;i++)
        hconcat(out,quarter_arr2[i], out);
    
    return out;
 
}
 
void to442_sobel(Mat in){
    Mat out = in;
    Mat dstx;
    filter2D(out, dstx, -1, kernelx);
    Mat dsty;
    filter2D(out, dsty, -1, kernely);
    abs(dstx);
    abs(dsty);
    add(dstx, dsty, out);
    
}
 
// I LEFT THIS THE SAME
void to442_greyscale(Mat in){
    Mat out = in;
    for (int i = 0; i < in.cols; i++){
        for (int j = 0; j < in.rows; j++){
            Vec3b greyscale_color = in.at<Vec3b>(Point(i, j));
            float grey_chan = 0.0722 * greyscale_color[0] + 0.2126 * 
                            greyscale_color[1] + 0.7152 * greyscale_color[2];
            greyscale_color[0] = grey_chan;
            greyscale_color[1] = grey_chan;
            greyscale_color[2] = grey_chan;
            in.at<Vec3b>(Point(i, j)) = greyscale_color;
        }
    }
}
