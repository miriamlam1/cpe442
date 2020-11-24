#include "opencv2/videoio.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h> 
#include <iostream>
#include <arm_neon.h>
#include <opencv2/imgproc.hpp>
#include <chrono>
 
using namespace cv;
using namespace std;
 
// init functions
void init_frames(VideoCapture vc);
void to442_greyscale(Mat frame);
void to442_sobel(Mat in, Mat out, uint16_t qcol, uint16_t qrow);
void* grey_and_sobel(Mat frame);
Mat divide_image(Mat frame);
 
#define WAIT 10
#define NUM_THREADS 4
#define BUFFER 1 // sobel overlap in pixels. should be 2
 
//static int y = 0;
uint16_t height;
 
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
 
typedef struct thread{
     Mat *in;
     Mat *out;
     int quarter;
 } ThreadFrame;
 
void init_frames(VideoCapture vc){
    Mat frame;
    String windowName = "Multi-Threaded Sobel";
    vc >> frame;
    height = frame.rows;
    int framecount = 0;
    cout << "For video of height: "<< height << " and width of: " << frame.cols << endl;
    auto start = std::chrono::high_resolution_clock::now();
    while (!frame.empty()){
     
        // Do stuff with frame
        frame = divide_image(frame);
 
        // Output frame
        imshow(windowName, frame);
 
        // waits WAIT ms and checks if escapekey is pressed to escape
        if (waitKey(WAIT) == 27)
            break;
            
        vc >> frame;
        framecount++;
    }
    
    auto finish = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish-start).count();
    cout<< "total frames: " << framecount << " total time(ms): " <<duration << endl;
    float avg_fps = (float(framecount)*1000)/float(duration);
    cout<< "average FPS: " << avg_fps << endl;
}
 
 int quartersize;
 
// each quarter runs this thread
void* grey_and_sobel(void* frame_ptr){
    
    ThreadFrame* mythreadframe = (ThreadFrame*) frame_ptr;
    Mat frame = (Mat)(*mythreadframe->in);
    Mat out = (Mat)(*mythreadframe->out);
    int quarter = mythreadframe->quarter;

    to442_greyscale(frame);
    
    uint16_t qcol = quarter*quartersize;
    to442_sobel(frame, out, qcol, height);
    
    pthread_exit(NULL); // exit because its done with thread at this point
}


 
Mat divide_image(Mat frame){
    int i; // counter
    Mat out = frame.clone();
    quartersize = frame.cols/NUM_THREADS;
    pthread_t threads[NUM_THREADS]; // list of threads
    ThreadFrame threadarray[NUM_THREADS];
 
    // init thread struct
    for(i=0; i<NUM_THREADS; i++){
        threadarray[i]=  {.in = &frame,
            .out = &out,
            .quarter = i};
    };
    
    // start the threads
    for(i=0; i<NUM_THREADS; i++){
        pthread_create (&threads[i], NULL, grey_and_sobel, &threadarray[i]);
    }

    // wait for all 4 threads to terminate
    for(int i=0;i<NUM_THREADS;i++){
        if(pthread_join(threads[i],NULL)<0)
            cout << "Thread join error" << endl;
    }
    
    return out;
 
}
 

// do not include the middle pixel in each since it is unchanged and need a vector of 16b
const int16_t Gx[] = {-1, 0, 1, -2, 2, -1, 0, 1};
                  const int16_t Gy[]=      {-1, -2, -1, 0, 0, 1, 2, 1};

void to442_sobel(Mat in, Mat out, uint16_t qcol, uint16_t qrow){
    for(int x = 1; x<in.cols-1; x++){
        for(int y = 1; y<in.rows-1; y++){
            int16_t Px = 0;
            int16_t Py = 0;
            int16_t a = in.at<Vec3b>(Point(x-1,y-1))[0]; // top left pixel
            int16_t b = in.at<Vec3b>(Point(x-1,y))[0]; // top middle pixel
            int16_t c = in.at<Vec3b>(Point(x-1,y+1))[0];
            int16_t d = in.at<Vec3b>(Point(x,y-1))[0];
            //int16_t mid = copy.at<Vec3b>(Point(x,y))[0];
            int16_t f = in.at<Vec3b>(Point(x,y+1))[0];
            int16_t g = in.at<Vec3b>(Point(x+1,y-1))[0];
            int16_t h = in.at<Vec3b>(Point(x+1,y))[0];
            int16_t i = in.at<Vec3b>(Point(x+1,y+1))[0];
            int16_t matrix[] = {a,b,c,d,f,g,h,i};
            int16x8_t f_vector = vld1q_s16(matrix);
            int16x8_t gx_vector = vld1q_s16(Gx);
            int16x8_t gy_vector = vld1q_s16(Gy);
            int16x8_t multx_vector = vmulq_s16(f_vector, gx_vector);
            int16x8_t multy_vector = vmulq_s16(f_vector, gy_vector);
            Px += vgetq_lane_s16(multx_vector,0);
            Px += vgetq_lane_s16(multx_vector,1);
            Px += vgetq_lane_s16(multx_vector,2);
            Px += vgetq_lane_s16(multx_vector,3);
            Px += vgetq_lane_s16(multx_vector,4);
            Px += vgetq_lane_s16(multx_vector,5);
            Px += vgetq_lane_s16(multx_vector,6);
            Px += vgetq_lane_s16(multx_vector,7);

            Py += vgetq_lane_s16(multy_vector,0);
            Py += vgetq_lane_s16(multy_vector,1);
            Py += vgetq_lane_s16(multy_vector,2);
            Py += vgetq_lane_s16(multy_vector,3);
            Py += vgetq_lane_s16(multy_vector,4);
            Py += vgetq_lane_s16(multy_vector,5);
            Py += vgetq_lane_s16(multy_vector,6);
            Py += vgetq_lane_s16(multy_vector,7);
            
            Px = abs(Px) + abs(Py);
            out.at<Vec3b>(Point(x,y)) = Vec3b(Px,Px,Px);
        }
    }
}
 
// I LEFT THIS THE SAME
void to442_greyscale(Mat in){
        float32_t blue_chan[4];
        float32_t green_chan[4];
        float32_t red_chan[4];
        float32_t grey_chan[4];

        float32_t red_scalar = 0.2126;
        float32_t blue_scalar = 0.7142;
        float32_t green_scalar = 0.0722;


        for (int i = 0; i < in.cols; i++){
                for(int j = 0; j < in.rows; j+=4){
                        for(int k = 0; k < 4 && j + k < in.rows; k++){
                                Vec3b pixel = in.at<Vec3b>(Point(i, (j + k)));
                                blue_chan[k] = pixel[0];
                                red_chan[k] = pixel[1];
                                green_chan[k] = pixel[2];
                        }

                        float32x4_t blue_vec = vld1q_f32(blue_chan);
                        float32x4_t green_vec = vld1q_f32(green_chan);
                        float32x4_t red_vec = vld1q_f32(red_chan);

                        red_vec = vmulq_n_f32(red_vec, red_scalar);
                        blue_vec = vmulq_n_f32(blue_vec, blue_scalar);
                        green_vec = vmulq_n_f32(green_vec, green_scalar);
                        
                        float32x4_t grey_vec;
                        grey_vec = vaddq_f32(red_vec, blue_vec);
                        grey_vec = vaddq_f32(grey_vec, green_vec);

                        vst1q_f32(grey_chan, grey_vec);
                        for(int k = 0; k < 4 && j + k < in.rows; k++){
                                in.at<Vec3b>(Point(i, (j +k))) = Vec3b(grey_chan[k], grey_chan[k], grey_chan[k]);
                        }
                }
        }
}
