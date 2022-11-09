#pragma once
#include "dl_layer_model.hpp"
#include "dl_layer_base.hpp"
#include "dl_layer_max_pool2d.hpp"
#include "dl_layer_conv2d.hpp"
#include "dl_layer_concat.hpp"
#include "handrecognition_coefficient.hpp"
#include "dl_layer_reshape.hpp"
#include "dl_layer_softmax.hpp"
// #include "dl_layer_transpose.hpp"
#include <stdint.h>

using namespace dl;
using namespace layer;
using namespace handrecognition_coefficient;

class HANDRECOGNITION : public Model<int16_t> // Derive the Model class in "dl_layer_model.hpp"
{
private:
    // Declare layers as member variables
    Reshape<int16_t> l1;
    Conv2D<int16_t> l2;
    MaxPool2D<int16_t> l3;
    Conv2D<int16_t> l4;
    MaxPool2D<int16_t> l5;
    Conv2D<int16_t> l6;
    MaxPool2D<int16_t> l7;
    //Transpose<int16_t> l8;
    Reshape<int16_t> l8;
    Conv2D<int16_t> l9;
    Conv2D<int16_t> l10;
    Conv2D<int16_t> l11;
    

public:
    Softmax<int16_t> l12; // a layer named l5_compress

    /**
     * @brief Initialize layers in constructor function
     * 
     */
    HANDRECOGNITION () : l1(Reshape<int16_t>({96,96,1})),
                         l2(Conv2D<int16_t>(-7, get_statefulpartitionedcall_sequential_conv2d_biasadd_filter(), get_statefulpartitionedcall_sequential_conv2d_biasadd_bias(), get_statefulpartitionedcall_sequential_conv2d_biasadd_activation(), PADDING_VALID, {}, 1,1, "l1")),
                         l3(MaxPool2D<int16_t>({2,2},PADDING_VALID, {}, 2, 2, "l2")),                      
                         l4(Conv2D<int16_t>(-6, get_statefulpartitionedcall_sequential_conv2d_1_biasadd_filter(), get_statefulpartitionedcall_sequential_conv2d_1_biasadd_bias(), get_statefulpartitionedcall_sequential_conv2d_1_biasadd_activation(), PADDING_VALID,{}, 1,1, "l3")),                       
                         l5(MaxPool2D<int16_t>({2,2},PADDING_VALID,{}, 2, 2, "l4")),                       
                         l6(Conv2D<int16_t>(-8, get_statefulpartitionedcall_sequential_conv2d_2_biasadd_filter(), get_statefulpartitionedcall_sequential_conv2d_2_biasadd_bias(), get_statefulpartitionedcall_sequential_conv2d_2_biasadd_activation(), PADDING_VALID,{}, 1,1, "l5")),                    
                         l7(MaxPool2D<int16_t>({2,2},PADDING_VALID,{}, 2, 2, "l6")),
                        //  l8(Transpose<int16_t>({},"17_transpose")),
                         l8(Reshape<int16_t>({1,1,1600},"l7_reshape")), //16,2id8,28 or 1,12544 or 12544, or 1,1,12544 or 1,16,28,28 or 1,16,28,28
                         l9(Conv2D<int16_t>(-8, get_fused_gemm_0_filter(), get_fused_gemm_0_bias(), get_fused_gemm_0_activation(), PADDING_VALID, {}, 1, 1, "l8")),
                         l10(Conv2D<int16_t>(-9, get_fused_gemm_1_filter(), get_fused_gemm_1_bias(), get_fused_gemm_1_activation(), PADDING_VALID, {}, 1, 1, "l9")),
                         l11(Conv2D<int16_t>(-9, get_fused_gemm_2_filter(), get_fused_gemm_2_bias(), NULL, PADDING_VALID,{}, 1,1, "l10")),
                         l12(Softmax<int16_t>(-14,"l11")){}

                
        
    /**
     * @brief call each layers' build(...) function in sequence
     * 
     * @param input 
     */
    void build(Tensor<int16_t> &input)
    {
        this->l1.build(input, true);
        this->l2.build(this->l1.get_output(), true);
        this->l3.build(this->l2.get_output(), true);
        this->l4.build(this->l3.get_output(), true);
        this->l5.build(this->l4.get_output(), true);
        this->l6.build(this->l5.get_output(), true);
        this->l7.build(this->l6.get_output(), true);
        this->l8.build(this->l7.get_output(), true);
        this->l9.build(this->l8.get_output(), true);
        this->l10.build(this->l9.get_output(), true);
        this->l11.build(this->l10.get_output(), true);
        this->l12.build(this->l11.get_output(), true);
      //  this->l13.build(this->l12.get_output(), true);
        
    }

    /**
     * @brief call each layers' call(...) function in sequence
     * 
     * @param input 
     */
    void call(Tensor<int16_t> &input)
    {
        this->l1.call(input);
        input.free_element();

        this->l2.call(this->l1.get_output());
        this->l1.get_output().free_element();

        this->l3.call(this->l2.get_output());
        this->l2.get_output().free_element();

        this->l4.call(this->l3.get_output());
        this->l3.get_output().free_element();

        this->l5.call(this->l4.get_output());
        this->l4.get_output().free_element();

        this->l6.call(this->l5.get_output());
        this->l5.get_output().free_element();

        this->l7.call(this->l6.get_output());
        this->l6.get_output().free_element();

        this->l8.call(this->l7.get_output());
        this->l7.get_output().free_element();

        this->l9.call(this->l8.get_output());
        this->l8.get_output().free_element();

        this->l10.call(this->l9.get_output());
        this->l9.get_output().free_element();

        this->l11.call(this->l10.get_output());
        this->l10.get_output().free_element();

        this->l12.call(this->l11.get_output());
        this->l11.get_output().free_element();

        // this->l13.call(this->l12.get_output());
        // this->l12.get_output().free_element();

    }
};




