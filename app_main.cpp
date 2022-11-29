#include "who_camera.h"
#include "model.hpp"
#include "model_define.hpp"
#include "who_lcd.h"


static QueueHandle_t xQueueAIFrame = NULL;
static QueueHandle_t xQueueLCDFrame = NULL;

extern "C" void app_main()
{
    xQueueAIFrame = xQueueCreate(2, sizeof(camera_fb_t *));
    xQueueLCDFrame = xQueueCreate(2, sizeof(camera_fb_t *));

    register_camera(PIXFORMAT_RGB565, FRAMESIZE_96X96, 2, xQueueAIFrame);//PIXFORMAT_GRAYSCALE, PIXFORMAT_RGB565
    register_hand_detection(xQueueAIFrame, NULL, NULL, xQueueLCDFrame, false);
    register_lcd(xQueueLCDFrame, NULL, true);
}

