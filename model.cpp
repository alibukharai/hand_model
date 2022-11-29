#include "model.hpp"
#include "model_define.hpp"
#include "esp_log.h"
#include "esp_camera.h"
#include <stdint.h>
#include "dl_image.hpp"


using namespace dl;
using namespace layer;
using namespace handrecognition_coefficient;



static const char *TAG = "hand_detection";

static QueueHandle_t xQueueFrameI = NULL;
static QueueHandle_t xQueueEvent = NULL;
static QueueHandle_t xQueueFrameO = NULL;
static QueueHandle_t xQueueResult = NULL;

static bool gEvent = true;
static bool gReturnFB = true;

static void task_process_handler(void *arg)
{
    camera_fb_t *frame = NULL;

    while (true)
    {
        if (gEvent)
        {
            bool is_detected = false;
            if (xQueueReceive(xQueueFrameI, &frame, portMAX_DELAY))
            {
                /*buffer to store pixel for gray scale*/
// printf("\n\n");

int input_height = 96;
int input_width = 96;
int input_channel = 1;
int input_exponent = -7;

                int16_t *model_input = (int16_t *)dl::tool::malloc_aligned_prefer(96*96*1, sizeof(int16_t *));

                
                for(int i=0 ;i<input_height*input_width*input_channel; i++){
                    
                    
                    model_input[i]= dl::image::convert_pixel_rgb565_to_gray(frame->buf[i]); 
                    // printf ("0x%02x,", model_input[i]);

                    // int16_t b =  (*((int16_t *)(frame->buf)+i) & 0x001F) >> 3;
                    // int16_t g = (*((int16_t *)(frame->buf)+i) & 0xE0) >> 10 | (*((int16_t *)(frame->buf)+i) & 0x0007) << 5;
                    // int16_t r = (*((int16_t *)(frame->buf)+i) & 0x00F8);

                    // int16_t grey =(r * 0.299 +g * 0.587 +b * 0.114);
                    // // printf("0x%02x,",grey); //%256);
                    // model_input[i] = grey;
                    
                    // // int normalized_input = frame->buf[i]; // 255.0; //normalization
                    // model_input[i] = (int16_t)DL_CLIP(grey * (1 << -input_exponent), -32768, 32767);
                    // printf("0x%02x,",model_input[i]); //%256);
                }        
        
                
// printf("\n\n");

                /* model call  */ 
                
                Tensor<int16_t> input;
               // input.set_element((int16_t *)frame->buf).set_exponent(-7).set_shape({96, 96, 1}).set_auto_free(false);
                input.set_element((int16_t *)model_input).set_exponent(input_exponent).set_shape({input_height,input_width,input_channel}).set_auto_free(false);
                HANDRECOGNITION model;
                dl::tool::Latency latency;
                latency.start();
                model.forward(input);
                latency.end();
                latency.print("SIGN", "forward");
                float *score = model.l11.get_output().get_element_ptr();
                // is_detected = true;
                float max_score = score[0];
                int max_index = 0;
                for (size_t i = 0; i < 10; i++)
                {
                    printf("%f, ", score[i]*100);
                    if (score[i] > max_score)
                    {
                        max_score = score[i];
                        max_index = i;
                    }
                }
                free(model_input);
                printf("\n");

                switch (max_index)
                {
                    case 0:
                    printf("Palm: 0");
                    break;
                    case 1:
                    printf("I: 1");
                    break;
                    case 2:
                    printf("fist: 2");
                    break;
                    case 3:
                    printf("fist_move: 3");
                    break;
                    case 4:
                    printf("thumb: 4");
                    break;
                    case 5:
                    printf("index: 5");
                    break;
                    case 6:
                    printf("ok: 6");
                    break;
                    case 7:
                    printf("palm move: 7");
                    break;
                    case 8:
                    printf("C: 8");
                    break;
                    case 9:
                    printf("down: 9");
                    break;
                    default:
                    printf("No result");

                }
                printf("\n");
                is_detected = true;

            }

            
    
            if (xQueueFrameO)
            {
                xQueueSend(xQueueFrameO, &frame, portMAX_DELAY);
            }
            else if (gReturnFB)
            {
                esp_camera_fb_return(frame);
            }
            else
            {
                free(frame);
            }

            if (xQueueResult)
            {
                xQueueSend(xQueueResult, &is_detected, portMAX_DELAY);
            }
            
        }
    }
}

static void task_event_handler(void *arg)
{
    while (true)
    {
        xQueueReceive(xQueueEvent, &(gEvent), portMAX_DELAY);
    }
}

void register_hand_detection(const QueueHandle_t frame_i,
                                 const QueueHandle_t event,
                                 const QueueHandle_t result,
                                 const QueueHandle_t frame_o,
                                 const bool camera_fb_return)
{
    xQueueFrameI = frame_i;
    xQueueFrameO = frame_o;
    xQueueEvent = event;
    xQueueResult = result;
    gReturnFB = camera_fb_return;

    xTaskCreatePinnedToCore(task_process_handler, TAG, 4 * 1024, NULL, 5, NULL, 0);
    if (xQueueEvent)
        xTaskCreatePinnedToCore(task_event_handler, TAG, 4 * 1024, NULL, 5, NULL, 1);
}
