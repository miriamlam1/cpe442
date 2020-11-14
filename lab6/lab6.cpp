#include "opencv2/videoio.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h> 
#include <iostream>
#include <cstdlib>
#include <arm_neon.h>
#include <stdlib.h>
#include "libperf.h"
#include <opencv2/imgproc.hpp>
#include <inttypes.h> /* for PRIu64 definition */
#include <chrono>
 #define NUM_THREADS 4
using namespace cv;
using namespace std;

static uint64_t total_cache_misses[NUM_THREADS];
static uint64_t total_cpu_cycles[NUM_THREADS];
// init functions
void init_frames(VideoCapture vc);
void to442_greyscale(Mat frame);
void to442_sobel(Mat in);
void* grey_and_sobel(Mat frame);
Mat divide_image(Mat frame);
 
#define WAIT 10

#define BUFFER 2 // sobel overlap in pixels. should be 2
 
static Mat kernelx = (Mat_<char>(3, 3) << -1, 0, 1,
                   -2, 0, 2,
                    - 1, 0, 1);
static Mat kernely = (Mat_<char>(3, 3) << 1, 2, 1,
                   0, 0, 0,
                   -1, -2, -1);
 
static int y = 0;
int height;

 
 typedef struct tf{
     Mat quarter;
     uint64_t cpu_cycles;
     uint64_t cache_misses;
 } ThreadFrame;
 
// main
int main(int argc, char **argv){
    if (argc < 2){
        cout << "ERROR: Not enough arguments\n Usage: ./lab6.exe myvideo.mp4" << endl;
        return EXIT_FAILURE;
	return -1;
    }
    
    VideoCapture vc(argv[1]);
    init_frames(vc);
    double fps = vc.get(CAP_PROP_FPS);
    cout << "FPS of original video : " << fps << endl;
    vc.release();
    
    return EXIT_SUCCESS;
}
 
void init_frames(VideoCapture vc){
    int frame_count = 1;
    Mat frame;
    String windowName = "Multi-Threaded Sobel";
    vc >> frame;
    height = frame.rows;
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
        frame_count ++;
    }
    // fps calculations
    auto finish = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish-start).count();
    cout<< "total frames: " << frame_count << " total time(ms): " <<duration << endl;
    float avg_fps = (float(frame_count)*1000)/float(duration);
    cout<< "average FPS: " << avg_fps << endl;
    for(int i=0;i<NUM_THREADS;i++){
        cout<< "core: " << i << " | cache misses: " <<total_cache_misses[i] << endl;
        cout<< "       " << " | cpu cycles:   " << total_cpu_cycles[i] << endl;
    }
}
 
// each quarter runs this thread
void* grey_and_sobel(void* frame_ptr){
    
    struct libperf_data *pd = libperf_initialize(-1, -1);
    libperf_enablecounter(pd, LIBPERF_COUNT_HW_CPU_CYCLES);
    libperf_enablecounter(pd, LIBPERF_COUNT_HW_CACHE_MISSES);
    
    ThreadFrame* mythreadframe = (ThreadFrame*) frame_ptr;
    Mat frame = mythreadframe->quarter;
    
    //filters
    to442_greyscale(frame);
    to442_sobel(frame);
    
    mythreadframe -> cpu_cycles = libperf_readcounter(pd, LIBPERF_COUNT_HW_CPU_CYCLES);
    mythreadframe -> cache_misses = libperf_readcounter(pd, LIBPERF_COUNT_HW_CACHE_MISSES);
    libperf_disablecounter(pd, LIBPERF_COUNT_HW_CPU_CYCLES);
    libperf_disablecounter(pd, LIBPERF_COUNT_HW_CACHE_MISSES);
    libperf_close(pd);
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
    
    ThreadFrame newThread[NUM_THREADS];
    
    for(i=0; i<NUM_THREADS; i++){
        newThread[i] = {.quarter = quarter_arr[i],
            .cpu_cycles = 0,
            .cache_misses = 0};
    }
    
    for(i=0; i<NUM_THREADS; i++){
        //~ newThread = (ThreadFrame*)malloc(sizeof(ThreadFrame));
        //~ newThread -> quarter = quarter_arr[i];
        //~ cout << "Tsdfasdfasdfr" << endl;
        pthread_create (&threads[i], NULL, grey_and_sobel, &newThread[i]);
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
        total_cpu_cycles[i] += newThread[i].cpu_cycles;
        total_cache_misses[i] += newThread[i].cache_misses;
    }

        
    // concatenate the images together
    Mat out;
    // hconcat(quarter_arr2, out); THIS COMMAND DOESNT WORK WHY
    hconcat(quarter_arr2[0],quarter_arr2[1], out);
    for(i=2;i<NUM_THREADS;i++)
        hconcat(out,quarter_arr2[i], out);
    
    return out;
 
}
 
// removed so we can do our own vector math
 /*
void to442_sobel(Mat in){
    Mat dstx;
    filter2D(in, dstx, -1, kernelx);
    Mat dsty;
    filter2D(in, dsty, -1, kernely);
    abs(dstx);
    abs(dsty);
    add(dstx, dsty, in);
}*/

// do not include the middle pixel in each since it is unchanged and need a vector of 16b
const int16_t Gx[] = {-1, 0, 1, -2, 2, -1, 0, 1};
const int16_t Gy[]=  {-1, -2, -1, 0, 0, 1, 2, 1};

void to442_sobel(Mat in){
    // copy to preserve original, in is now out
    Mat copy = in.clone();

    for(int x = 1; x<in.cols-1; x++){
        for(int y = 1; y<in.rows-1; y++){
            int16_t Px = 0;
            int16_t Py = 0;
            int16_t a = copy.at<Vec3b>(Point(x-1,y-1))[0]; // top left pixel
            int16_t b = copy.at<Vec3b>(Point(x-1,y))[0]; // top middle pixel
            int16_t c = copy.at<Vec3b>(Point(x-1,y+1))[0];
            int16_t d = copy.at<Vec3b>(Point(x,y-1))[0];
            //int16_t mid = copy.at<Vec3b>(Point(x,y))[0];
            int16_t f = copy.at<Vec3b>(Point(x,y+1))[0];
            int16_t g = copy.at<Vec3b>(Point(x+1,y-1))[0];
            int16_t h = copy.at<Vec3b>(Point(x+1,y))[0];
            int16_t i = copy.at<Vec3b>(Point(x+1,y+1))[0];
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
            in.at<Vec3b>(Point(x,y)) = Vec3b(Px,Px,Px);
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


        // sizeof(float) == 4
        // sizeof(double) == 8

        Mat out = in;

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
