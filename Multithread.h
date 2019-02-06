#ifndef MULTITHREAD_H
#define MULTITHREAD_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <time.h>
#include <iostream>
#include <sstream> //this header file is needed when using stringstream
#include <fstream>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

using namespace std;

#define MNIST_TESTING_SET_IMAGE_FILE_NAME "data/t10k-images-idx3-ubyte"  ///< MNIST image testing file in the data folder
#define MNIST_TESTING_SET_LABEL_FILE_NAME "data/t10k-labels-idx1-ubyte"  ///< MNIST label testing file in the data folder

#define HIDDEN_WEIGHTS_FILE "net_params/hidden_weights.txt"
#define HIDDEN_BIASES_FILE "net_params/hidden_biases.txt"
#define OUTPUT_WEIGHTS_FILE "net_params/out_weights.txt"
#define OUTPUT_BIASES_FILE "net_params/out_biases.txt"

#define NUMBER_OF_INPUT_CELLS 784   ///< use 28*28 input cells (= number of pixels per MNIST image)
#define NUMBER_OF_HIDDEN_CELLS 256   ///< use 256 hidden cells in one hidden layer
#define NUMBER_OF_OUTPUT_CELLS 10   ///< use 10 output cells to model 10 digits (0-9)

#define MNIST_MAX_TESTING_IMAGES 10000                      ///< number of images+labels in the TEST file/s
#define MNIST_IMG_WIDTH 28                                  ///< image width in pixel
#define MNIST_IMG_HEIGHT 28                                 ///< image height in pixel

typedef struct MNIST_ImageFileHeader MNIST_ImageFileHeader;
typedef struct MNIST_LabelFileHeader MNIST_LabelFileHeader;

typedef struct MNIST_Image MNIST_Image;
typedef uint8_t MNIST_Label;
typedef struct Hidden_Node Hidden_Node;
typedef struct Output_Node Output_Node;

struct Hidden_Node{
    double weights[28*28];
    double bias;
    double output;
};

struct Output_Node{
    double weights[256];
    double bias;
    double output;
};

struct MNIST_Image{
    uint8_t pixel[28*28];
};

struct MNIST_ImageFileHeader{
    uint32_t magicNumber;
    uint32_t maxImages;
    uint32_t imgWidth;
    uint32_t imgHeight;
};

struct MNIST_LabelFileHeader{
    uint32_t magicNumber;
    uint32_t maxImages;
};

Hidden_Node hidden_nodes[NUMBER_OF_HIDDEN_CELLS];
Output_Node output_nodes[NUMBER_OF_OUTPUT_CELLS];

// number of incorrect predictions
int errCount = 0;

FILE *imageFile, *labelFile;
MNIST_Image img;
MNIST_Label lbl;
int MIDDLE_LAYER_THREAD_NUMBER;
#define OUTPUT_LAYER_THREAD_NUMBER 10

//semaphore
sem_t *input_layer_sem, *middle_input_layer_sem , *middle_output_layer_sem;
sem_t output_middle_layer_sem[OUTPUT_LAYER_THREAD_NUMBER];
sem_t output_result_layer_sem , result_layer_sem;
sem_t display_sem1 , display_sem2;


#endif