#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include "image.h"

#include <pthread.h> 
#define numThreads 8
//Pthreads stuff^^
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//An array of kernel matrices to be used for image convolution.  
//The indexes of these match the enumeration from the header file. ie. algorithms[BLUR] returns the kernel corresponding to a box blur.
Matrix algorithms[]={
    {{0,-1,0},{-1,4,-1},{0,-1,0}},
    {{0,-1,0},{-1,5,-1},{0,-1,0}},
    {{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0}},
    {{1.0/16,1.0/8,1.0/16},{1.0/8,1.0/4,1.0/8},{1.0/16,1.0/8,1.0/16}},
    {{-2,-1,0},{-1,1,1},{0,1,2}},
    {{0,0,0},{0,1,0},{0,0,0}}
};

typedef struct {
    Image* srcImage;
    Image* destImage;
    Matrix algorithm;
    int start_row;
    int end_row;
} ThreadData;
//data to each


//getPixelValue - Computes the value of a speciafic pixel on a specific channel using the selected convolution kernel
//Paramters: srcImage:  An Image struct populated with the image being convoluted
//           x: The x coordinate of the pixel
//          y: The y coordinate of the pixel
//          bit: The color channel being manipulated
//          algorithm: The 3x3 kernel matrix to use for the convolution
//Returns: The new value for this x,y pixel and bit channel
uint8_t getPixelValue(Image* srcImage,int x,int y,int bit,Matrix algorithm){
    int px,mx,py,my,i,span;
    span=srcImage->width*srcImage->bpp;
    // for the edge pixes, just reuse the edge pixel
    px=x+1; py=y+1; mx=x-1; my=y-1;
    if (mx<0) mx=0;
    if (my<0) my=0;
    if (px>=srcImage->width) px=srcImage->width-1;
    if (py>=srcImage->height) py=srcImage->height-1;
    uint8_t result=
        algorithm[0][0]*srcImage->data[Index(mx,my,srcImage->width,bit,srcImage->bpp)]+
        algorithm[0][1]*srcImage->data[Index(x,my,srcImage->width,bit,srcImage->bpp)]+
        algorithm[0][2]*srcImage->data[Index(px,my,srcImage->width,bit,srcImage->bpp)]+
        algorithm[1][0]*srcImage->data[Index(mx,y,srcImage->width,bit,srcImage->bpp)]+
        algorithm[1][1]*srcImage->data[Index(x,y,srcImage->width,bit,srcImage->bpp)]+
        algorithm[1][2]*srcImage->data[Index(px,y,srcImage->width,bit,srcImage->bpp)]+
        algorithm[2][0]*srcImage->data[Index(mx,py,srcImage->width,bit,srcImage->bpp)]+
        algorithm[2][1]*srcImage->data[Index(x,py,srcImage->width,bit,srcImage->bpp)]+
        algorithm[2][2]*srcImage->data[Index(px,py,srcImage->width,bit,srcImage->bpp)];
    return result;
}

//convolute:  Applies a kernel matrix to an image
//Parameters: srcImage: The image being convoluted
//            destImage: A pointer to a  pre-allocated (including space for the pixel array) structure to receive the convoluted image.  It should be the same size as srcImage
//            algorithm: The kernel matrix to use for the convolution
//Returns: Nothing

void* Myconvolute(void* arg) {
    ThreadData* data = (ThreadData*) arg;
    int row, pix, bit;

    // Loop ONLY over the rows assigned to this thread
    for (row = data->start_row; row < data->end_row; row++) {
        for (pix = 0; pix < data->srcImage->width; pix++) {
            for (bit = 0; bit < data->srcImage->bpp; bit++) {
                data->destImage->data[Index(pix, row, data->srcImage->width, bit, data->srcImage->bpp)] = 
                    getPixelValue(data->srcImage, pix, row, bit, data->algorithm);
            }
        }
    }
    return NULL;
} // What each thread run

//Usage: Prints usage information for the program
//Returns: -1
int Usage(){
    printf("Usage: image <filename> <type>\n\twhere type is one of (edge,sharpen,blur,gauss,emboss,identity)\n");
    return -1;
}

//GetKernelType: Converts the string name of a convolution into a value from the KernelTypes enumeration
//Parameters: type: A string representation of the type
//Returns: an appropriate entry from the KernelTypes enumeration, defaults to IDENTITY, which does nothing but copy the image.
enum KernelTypes GetKernelType(char* type){
    if (!strcmp(type,"edge")) return EDGE;
    else if (!strcmp(type,"sharpen")) return SHARPEN;
    else if (!strcmp(type,"blur")) return BLUR;
    else if (!strcmp(type,"gauss")) return GAUSE_BLUR;
    else if (!strcmp(type,"emboss")) return EMBOSS;
    else return IDENTITY;
}

//main:
//argv is expected to take 2 arguments.  First is the source file name (can be jpg, png, bmp, tga).  Second is the lower case name of the algorithm.
int main(int argc,char** argv){
    stbi_set_flip_vertically_on_load(0); 
    if (argc!=3) return Usage();
    char* fileName=argv[1];
    if (!strcmp(argv[1],"pic4.jpg")&&!strcmp(argv[2],"gauss")){
        printf("You have applied a gaussian filter to Gauss which has caused a tear in the time-space continum.\n");
    }
    enum KernelTypes type=GetKernelType(argv[2]);

    Image srcImage,destImage,bwImage;   
    srcImage.data=stbi_load(fileName,&srcImage.width,&srcImage.height,&srcImage.bpp,0);
    if (!srcImage.data){
        printf("Error loading file %s.\n",fileName);
        return -1;
    }
    destImage.bpp=srcImage.bpp;
    destImage.height=srcImage.height;
    destImage.width=srcImage.width;
    destImage.data=malloc(sizeof(uint8_t)*destImage.width*destImage.bpp*destImage.height);
    if (!destImage.data) {
        printf("Error allocating memory for destination image.\n");
        stbi_image_free(srcImage.data);
        return -1;
    }

    struct timespec start, end;
    double time_spent;
    clock_gettime(CLOCK_MONOTONIC, &start);
    pthread_t threads[numThreads];
    ThreadData thread_data[numThreads];

    // Calculate how many rows each thread gets
    int rows_per_thread_base = srcImage.height / numThreads;
    int remainder_rows = srcImage.height % numThreads;
    int current_start_row = 0;

    printf("Starting convolution with %d threads.\n", numThreads);

    for (int i = 0; i < numThreads; i++) {
        int rows_for_this_thread = rows_per_thread_base;
        //splitting work
        if (i < remainder_rows) {
            rows_for_this_thread++;
        }

        // Set up for thread
        thread_data[i].srcImage = &srcImage;
        thread_data[i].destImage = &destImage;
        memcpy(thread_data[i].algorithm, algorithms[type], sizeof(Matrix));
        thread_data[i].start_row = current_start_row;
        thread_data[i].end_row = current_start_row + rows_for_this_thread;
        
        current_start_row = thread_data[i].end_row;

        if (rows_for_this_thread > 0) {
            if (pthread_create(&threads[i], NULL, Myconvolute, &thread_data[i]) != 0) {
                perror("Failed to create thread");
                stbi_image_free(srcImage.data);
                free(destImage.data);
                return -1;
            }
        }
    }

    for (int i = 0; i < numThreads; i++) {
    //make sure thread made
        int rows_for_this_thread = (i < remainder_rows) ? rows_per_thread_base + 1 : rows_per_thread_base;
        if (rows_for_this_thread > 0) {
            pthread_join(threads[i], NULL);
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    
    printf("Took %f seconds\n", time_spent);

    stbi_write_png("output.png", destImage.width, destImage.height, destImage.bpp, destImage.data, destImage.bpp * destImage.width);
    free(destImage.data);
    stbi_image_free(srcImage.data);

   return 0;
}