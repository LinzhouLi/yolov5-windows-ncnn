#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "ncnn/net.h"
#include <iostream>

struct Object {
    float x;
    float y;
    float w;
    float h;
    int label;
    float prob;
};

static inline float intersection_area(const Object& a, const Object& b) {
    if (a.x > b.x + b.w || a.x + a.w < b.x || a.y > b.y + b.h || a.y + a.h < b.y) {
        // no intersection
        return 0.f;
    }

    float inter_width = min(a.x + a.w, b.x + b.w) - max(a.x, b.x);
    float inter_height = min(a.y + a.h, b.y + b.h) - max(a.y, b.y);

    return inter_width * inter_height;
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j) {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects) {
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked,
    float nms_threshold) {
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        //        areas[i] = faceobjects[i].rect.area();
        areas[i] = faceobjects[i].w * faceobjects[i].h;
    }

    for (int i = 0; i < n; i++) {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x) {
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad,
    const ncnn::Mat& feat_blob, float prob_threshold,
    std::vector<Object>& objects) {
    const int num_grid_x = feat_blob.w;
    const int num_grid_y = feat_blob.h;

    const int num_anchors = anchors.w / 2;

    const int num_class = feat_blob.c / num_anchors - 5;

    const int feat_offset = num_class + 5;

    for (int q = 0; q < num_anchors; q++) {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        for (int i = 0; i < num_grid_y; i++) {
            for (int j = 0; j < num_grid_x; j++) {
                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++) {
                    //                    float score = sigmoid(feat_blob.channel(q * feat_offset + 5 + k).row(i)[j]);
                    float score = feat_blob.channel(q * feat_offset + 5 + k).row(i)[j];

                    if (score > class_score) {
                        class_index = k;
                        class_score = score;
                    }
                }

                float box_score = feat_blob.channel(q * feat_offset + 4).row(i)[j];

                float confidence = sigmoid(box_score) * sigmoid(class_score);
                //                float confidence = box_score * class_score;
                if (confidence >= prob_threshold) {

                    float dx = sigmoid(feat_blob.channel(q * feat_offset + 0).row(i)[j]);
                    float dy = sigmoid(feat_blob.channel(q * feat_offset + 1).row(i)[j]);
                    float dw = sigmoid(feat_blob.channel(q * feat_offset + 2).row(i)[j]);
                    float dh = sigmoid(feat_blob.channel(q * feat_offset + 3).row(i)[j]);

                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    Object obj;
                    obj.x = x0;
                    obj.y = y0;
                    obj.w = x1 - x0;
                    obj.h = y1 - y0;
                    obj.label = class_index;
                    obj.prob = confidence;


                    objects.push_back(obj);
                }
            }
        }
    }
}

void printMatInfo(const ncnn::Mat& mat) {
    std::cout << "dims: " << mat.dims
        << " w: " << mat.w
        << " h: " << mat.h
        << " c: " << mat.c
        << std::endl;
}

void printMat(const ncnn::Mat& mat)
{
    printf("\n");
    for (int q = 0; q < mat.c; q++) {
        const float* ptr = mat.channel(q);
        for (int y = 0; y < mat.h; y++) {
            for (int x = 0; x < mat.w; x++)
                printf("%f ", ptr[x]);
            ptr += mat.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
}

int main() {

    // 读取图片
	cv::Mat img = cv::imread("D:\\CODE\\C++\\yolov5ncnn\\2.jpg");
	const int height = img.rows;
	const int width = img.cols;
	std::cout << "width:" << width << "\nheight:" << height << std::endl;

    // 读取模型
    ncnn::Net yolov5;
    yolov5.load_param("D:\\CODE\\C++\\yolov5ncnn\\yolov5n.ncnn.param");
    yolov5.load_model("D:\\CODE\\C++\\yolov5ncnn\\yolov5n.ncnn.bin");

    // 设置输入图片resize后大小
    const int target_size = 640;
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h) {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    } else {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }
    std::cout << "\nresized width: " << w << "\nresized height: " << h << std::endl;

    // 输入矩阵
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, width, height, w, h);

    // pad
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f); // 以114.0常量填充
    std::cout << "\nin pad\n";
    printMatInfo(in_pad);

    // [0, 255] -> [0, 1]
    const float norm[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    in_pad.substract_mean_normalize(0, norm);

    // yolov5
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;

    std::vector<Object> proposals;

    ncnn::Extractor ex = yolov5.create_extractor();
    ex.input("in0", in_pad);
    
    // stride 8
    {
        ncnn::Mat out;
        ex.extract("out0", out); // output
        std::cout << "\nstride 8" << std::endl;
        printMatInfo(out);

        ncnn::Mat anchors(6);
        anchors[0] = 10.f;
        anchors[1] = 13.f;
        anchors[2] = 16.f;
        anchors[3] = 30.f;
        anchors[4] = 33.f;
        anchors[5] = 23.f;

        std::vector<Object> objects8;
        generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);
        proposals.insert(proposals.end(), objects8.begin(), objects8.end());

    }

    // stride 16
    {
        ncnn::Mat out;
        ex.extract("out1", out); // 781
        std::cout << "\nstride 16" << std::endl;
        printMatInfo(out);

        ncnn::Mat anchors(6);
        anchors[0] = 30.f;
        anchors[1] = 61.f;
        anchors[2] = 62.f;
        anchors[3] = 45.f;
        anchors[4] = 59.f;
        anchors[5] = 119.f;

        std::vector<Object> objects16;
        generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);
        proposals.insert(proposals.end(), objects16.begin(), objects16.end());

    }

    // stride 32
    {
        ncnn::Mat out;
        ex.extract("out2", out); // 801
        std::cout << "\nstride 32" << std::endl;
        printMatInfo(out);
        //printMat(out);

        ncnn::Mat anchors(6);
        anchors[0] = 116.f;
        anchors[1] = 90.f;
        anchors[2] = 156.f;
        anchors[3] = 198.f;
        anchors[4] = 373.f;
        anchors[5] = 326.f;

        std::vector<Object> objects32;
        generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);
        proposals.insert(proposals.end(), objects32.begin(), objects32.end());

    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    std::vector<Object> objects;
    int count = picked.size();
    objects.resize(count);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].x - (wpad / 2)) / scale;
        float y0 = (objects[i].y - (hpad / 2)) / scale;
        float x1 = (objects[i].x + objects[i].w - (wpad / 2)) / scale;
        float y1 = (objects[i].y + objects[i].h - (hpad / 2)) / scale;

        // clip
        x0 = max(min(x0, (float)(width - 1)), 0.f);
        y0 = max(min(y0, (float)(height - 1)), 0.f);
        x1 = max(min(x1, (float)(width - 1)), 0.f);
        y1 = max(min(y1, (float)(height - 1)), 0.f);

        objects[i].x = x0;
        objects[i].y = y0;
        objects[i].w = x1 - x0;
        objects[i].h = y1 - y0;
    }

    for (Object &obj : objects) {
        std::cout << "\nlabel: " << obj.label
            << "\nprob: " << obj.prob 
            << "\nxy: " << obj.x << "," << obj.y
            << "\nwh: " << obj.w << "," << obj.h 
            << std::endl;
        cv::rectangle(
            img,
            cv::Rect(obj.x, obj.y, obj.w, obj.h),
            cv::Scalar(255, 0, 0),
            5
        );
    }

    const float scale2 = 0.3f;
    cv::Mat img_show; // 等比例缩小图
    resize(img, img_show, cv::Size(width * scale2, height * scale2)); // 缩小操作
    cv::imshow("PlamDetect", img_show);
    cv::waitKey(0);

	return 0;

}