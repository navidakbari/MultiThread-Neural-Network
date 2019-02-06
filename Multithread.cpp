#include "Multithread.h"

void locateCursor(const int row, const int col){
    printf("%c[%d;%dH",27,row,col);
}

void clearScreen(){
    printf("\e[1;1H\e[2J");
}

void displayImage(MNIST_Image *img, int row, int col){

    char imgStr[(MNIST_IMG_HEIGHT * MNIST_IMG_WIDTH)+((col+1)*MNIST_IMG_HEIGHT)+1];
    strcpy(imgStr, "");

    for (int y=0; y<MNIST_IMG_HEIGHT; y++){

        for (int o=0; o<col-2; o++) strcat(imgStr," ");
        strcat(imgStr,"|");

        for (int x=0; x<MNIST_IMG_WIDTH; x++){
            strcat(imgStr, img->pixel[y*MNIST_IMG_HEIGHT+x] ? "X" : "." );
        }
        strcat(imgStr,"\n");
    }

    if (col!=0 && row!=0) locateCursor(row, 0);
    printf("%s",imgStr);
}

void displayImageFrame(int row, int col){

    if (col!=0 && row!=0) locateCursor(row, col);

    printf("------------------------------\n");

    for (int i=0; i<MNIST_IMG_HEIGHT; i++){
        for (int o=0; o<col-1; o++) printf(" ");
        printf("|                            |\n");
    }

    for (int o=0; o<col-1; o++) printf(" ");
    printf("------------------------------");

}

void displayLoadingProgressTesting(int imgCount, int y, int x){

    float progress = (float)(imgCount+1)/(float)(MNIST_MAX_TESTING_IMAGES)*100;

    if (x!=0 && y!=0) locateCursor(y, x);

    printf("Testing image No. %5d of %5d images [%d%%]\n                                  ",(imgCount+1),MNIST_MAX_TESTING_IMAGES,(int)progress);

}

void displayProgress(int imgCount, int errCount, int y, int x){

    double successRate = 1 - ((double)errCount/(double)(imgCount+1));

    if (x!=0 && y!=0) locateCursor(y, x);

    printf("Result: Correct=%5d  Incorrect=%5d  Success-Rate= %5.2f%% \n",imgCount+1-errCount, errCount, successRate*100);


}

uint32_t flipBytes(uint32_t n){

    uint32_t b0,b1,b2,b3;

    b0 = (n & 0x000000ff) <<  24u;
    b1 = (n & 0x0000ff00) <<   8u;
    b2 = (n & 0x00ff0000) >>   8u;
    b3 = (n & 0xff000000) >>  24u;

    return (b0 | b1 | b2 | b3);

}

void readImageFileHeader(FILE *imageFile, MNIST_ImageFileHeader *ifh){

    ifh->magicNumber =0;
    ifh->maxImages   =0;
    ifh->imgWidth    =0;
    ifh->imgHeight   =0;

    fread(&ifh->magicNumber, 4, 1, imageFile);
    ifh->magicNumber = flipBytes(ifh->magicNumber);

    fread(&ifh->maxImages, 4, 1, imageFile);
    ifh->maxImages = flipBytes(ifh->maxImages);

    fread(&ifh->imgWidth, 4, 1, imageFile);
    ifh->imgWidth = flipBytes(ifh->imgWidth);

    fread(&ifh->imgHeight, 4, 1, imageFile);
    ifh->imgHeight = flipBytes(ifh->imgHeight);
}

void readLabelFileHeader(FILE *imageFile, MNIST_LabelFileHeader *lfh){

    lfh->magicNumber =0;
    lfh->maxImages   =0;

    fread(&lfh->magicNumber, 4, 1, imageFile);
    lfh->magicNumber = flipBytes(lfh->magicNumber);

    fread(&lfh->maxImages, 4, 1, imageFile);
    lfh->maxImages = flipBytes(lfh->maxImages);

}

FILE *openMNISTImageFile(char *fileName){

    FILE *imageFile;
    imageFile = fopen (fileName, "rb");
    if (imageFile == NULL) {
        printf("Abort! Could not fine MNIST IMAGE file: %s\n",fileName);
        exit(0);
    }

    MNIST_ImageFileHeader imageFileHeader;
    readImageFileHeader(imageFile, &imageFileHeader);

    return imageFile;
}

FILE *openMNISTLabelFile(char *fileName){

    FILE *labelFile;
    labelFile = fopen (fileName, "rb");
    if (labelFile == NULL) {
        printf("Abort! Could not find MNIST LABEL file: %s\n",fileName);
        exit(0);
    }

    MNIST_LabelFileHeader labelFileHeader;
    readLabelFileHeader(labelFile, &labelFileHeader);

    return labelFile;
}

MNIST_Image getImage(FILE *imageFile){

    MNIST_Image img;
    size_t result;
    result = fread(&img, sizeof(img), 1, imageFile);
    if (result!=1) {
        printf("\nError when reading IMAGE file! Abort!\n");
        exit(1);
    }

    return img;
}

MNIST_Label getLabel(FILE *labelFile){
    MNIST_Label lbl;
    size_t result;
    result = fread(&lbl, sizeof(lbl), 1, labelFile);
    if (result!=1) {
        printf("\nError when reading LABEL file! Abort!\n");
        exit(1);
    }

    return lbl;
}

void allocateHiddenParameters(){
    int idx = 0;
    int bidx = 0;
    ifstream weights(HIDDEN_WEIGHTS_FILE);
    for(string line; getline(weights, line); )   //read stream line by line
    {
        stringstream in(line);
        for (int i = 0; i < 28*28; ++i){
            in >> hidden_nodes[idx].weights[i];
      }
      idx++;
    }
    weights.close();

    ifstream biases(OUTPUT_BIASES_FILE);
    for(string line; getline(biases, line); )   //read stream line by line
    {
        stringstream in(line);
        in >> hidden_nodes[bidx].bias;
        bidx++;
    }
    biases.close();

}

void allocateOutputParameters(){
    int idx = 0;
    int bidx = 0;
    ifstream weights(OUTPUT_WEIGHTS_FILE); //"layersinfo.txt"
    for(string line; getline(weights, line); )   //read stream line by line
    {
        stringstream in(line);
        for (int i = 0; i < 256; ++i){
            in >> output_nodes[idx].weights[i];
      }
      idx++;
    }
    weights.close();

    ifstream biases(OUTPUT_BIASES_FILE);
    for(string line; getline(biases, line); )   //read stream line by line
    {
        stringstream in(line);
        in >> output_nodes[bidx].bias;
        bidx++;
    }
    biases.close();

}

int getNNPrediction(){

    double maxOut = 0;
    int maxInd = 0;

    for (int i=0; i<NUMBER_OF_OUTPUT_CELLS; i++){

        if (output_nodes[i].output > maxOut){
            maxOut = output_nodes[i].output;
            maxInd = i;
        }
    }

    return maxInd;

}

void* input_layer(void* param){

    for (int imgCount = 0; imgCount < MNIST_MAX_TESTING_IMAGES; imgCount++){
        for(int i = 0 ; i < MIDDLE_LAYER_THREAD_NUMBER ; i++)
            sem_wait(&middle_input_layer_sem[i]);
        
        sem_wait(&display_sem1);

        displayLoadingProgressTesting(imgCount,5,5);
        img = getImage(imageFile);
        displayImage(&img, 8,6);
        
        sem_post(&display_sem2);
        for(int i = 0 ; i < MIDDLE_LAYER_THREAD_NUMBER ; i++)
            sem_post(&input_layer_sem[i]);
    }
    fclose(imageFile);
}

void* middle_layer(void* param){

    int thread_id = *(int *) param;

    for (int imgCount=0; imgCount<MNIST_MAX_TESTING_IMAGES; imgCount++){
        sem_wait(&input_layer_sem[thread_id]); 

        for(int i = 0 ; i < OUTPUT_LAYER_THREAD_NUMBER ; i++)
            sem_wait(&middle_output_layer_sem[thread_id]);

        int temp = NUMBER_OF_HIDDEN_CELLS / MIDDLE_LAYER_THREAD_NUMBER;

        for (int j = 0 + (thread_id) * temp; j < (thread_id) * temp + temp; j++) {
            hidden_nodes[j].output = 0;
            for (int z = 0; z < NUMBER_OF_INPUT_CELLS; z++) {
                hidden_nodes[j].output += img.pixel[z] * hidden_nodes[j].weights[z];
            }
            hidden_nodes[j].output += hidden_nodes[j].bias;
            hidden_nodes[j].output = (hidden_nodes[j].output >= 0) ?  hidden_nodes[j].output : 0;
        }

        sem_post(&middle_input_layer_sem[thread_id]);
        for(int i = 0 ; i < OUTPUT_LAYER_THREAD_NUMBER ; i++){
            sem_post(&output_middle_layer_sem[i]);
        }
    }
}

void* output_layer(void* param){
    int thread_id = *((int *) param);

    for (int imgCount = 0 ; imgCount < MNIST_MAX_TESTING_IMAGES ; imgCount++){
        for(int i = 0 ; i < MIDDLE_LAYER_THREAD_NUMBER ; i++)
            sem_wait(&output_middle_layer_sem[thread_id]);
        sem_wait(&result_layer_sem);
        
        output_nodes[thread_id].output = 0;
        for (int j = 0; j < NUMBER_OF_HIDDEN_CELLS; j++) {
            output_nodes[thread_id].output += hidden_nodes[j].output * output_nodes[thread_id].weights[j];
        }
        output_nodes[thread_id].output += 1/(1+ exp(-1* output_nodes[thread_id].output));
        
        sem_post(&output_result_layer_sem);
        for(int i = 0 ; i < MIDDLE_LAYER_THREAD_NUMBER ; i++)
            sem_post(&middle_output_layer_sem[i]);
    }
}

void* result_layer(void* param){
    for (int imgCount = 0; imgCount < MNIST_MAX_TESTING_IMAGES; imgCount++){
        for(int i = 0 ; i < OUTPUT_LAYER_THREAD_NUMBER ; i++)
            sem_wait(&output_result_layer_sem);

        lbl = getLabel(labelFile);

        int predictedNum = getNNPrediction();
        if (predictedNum!=lbl) errCount++;

        sem_wait(&display_sem2);
        printf("\n      Prediction: %d   Actual: %d ",predictedNum, lbl);
        displayProgress(imgCount, errCount, 5, 66);
        sem_post(&display_sem1);
        
        for(int i = 0 ; i < OUTPUT_LAYER_THREAD_NUMBER ; i++)
            sem_post(&result_layer_sem);
    }
    fclose(labelFile);
}

void testNN(){

    imageFile = openMNISTImageFile(MNIST_TESTING_SET_IMAGE_FILE_NAME);
    labelFile = openMNISTLabelFile(MNIST_TESTING_SET_LABEL_FILE_NAME);
    int i = 0;

    displayImageFrame(7,5);
    
    //Semaphore
    for(int j = 0 ; j < MIDDLE_LAYER_THREAD_NUMBER ; j++){
        sem_init(&input_layer_sem[j] , 0 ,  0);
        sem_init(&middle_input_layer_sem[j] , 0 , 1);
        sem_init(&middle_output_layer_sem[j] , 0 , 10);
    }
    
    for(int j = 0 ; j < OUTPUT_LAYER_THREAD_NUMBER ; j++)
        sem_init(&output_middle_layer_sem[j] , 0 , 0);
    
    sem_init(&output_result_layer_sem , 0 , 0);
    sem_init(&result_layer_sem , 0 , 10);
    sem_init(&display_sem1 , 0 , 1);
    sem_init(&display_sem2 , 0 , 0);

    
    pthread_t input_layer_thread;
    pthread_create(&input_layer_thread , NULL , input_layer , (void *)(intptr_t)i);

    pthread_t middle_layer_thread[MIDDLE_LAYER_THREAD_NUMBER];

    for(int j = 0 ; j < MIDDLE_LAYER_THREAD_NUMBER ; j++){
        int *arg = (int*)malloc(sizeof(int));
        *arg = j;
        pthread_create(&middle_layer_thread[j] , NULL , middle_layer ,(void*)(arg));
    }
    
    pthread_t output_layer_thread[OUTPUT_LAYER_THREAD_NUMBER];
    for(int j = 0 ; j < OUTPUT_LAYER_THREAD_NUMBER ; j++){  
        int *arg = (int*)malloc(sizeof(*arg));
        *arg = j;
        pthread_create(&output_layer_thread[j] , NULL , output_layer , arg);
    }

    pthread_t result_layer_thread;
    pthread_create(&result_layer_thread , NULL , result_layer , (void *)(intptr_t)i);

    pthread_join(input_layer_thread  , NULL);
    for(int j = 0 ; j < MIDDLE_LAYER_THREAD_NUMBER ; j++)
        pthread_join(middle_layer_thread[j] , NULL);
    for(int j = 0 ; j < OUTPUT_LAYER_THREAD_NUMBER ; j++)
        pthread_join(output_layer_thread[j] , NULL);
    pthread_join(result_layer_thread , NULL);

}

void initialize_semaphore(){
    input_layer_sem = (sem_t *)malloc(sizeof(sem_t) * MIDDLE_LAYER_THREAD_NUMBER);
    middle_input_layer_sem = (sem_t *)malloc(sizeof(sem_t) * MIDDLE_LAYER_THREAD_NUMBER);
    middle_output_layer_sem = (sem_t *)malloc(sizeof(sem_t) * MIDDLE_LAYER_THREAD_NUMBER);

}

int main(int argc, const char * argv[]) {

    cout << "Number of Thread? " << endl;
    cin >> MIDDLE_LAYER_THREAD_NUMBER;
    initialize_semaphore();

    time_t startTime = time(NULL);
    clearScreen();
    printf("    MNIST-NN: a simple 2-layer neural network processing the MNIST handwriting images\n");

    allocateHiddenParameters();
    allocateOutputParameters();

    testNN();

    locateCursor(38, 5);

    time_t endTime = time(NULL);
    double executionTime = difftime(endTime, startTime);
    printf("\n    DONE! Total execution time: %.1f sec\n\n",executionTime);
    return 0;
}