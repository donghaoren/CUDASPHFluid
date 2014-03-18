#include "videomaker.h"
#include <stdio.h>
#include <string>

//#define cimg_display 0

#include "bitmap_image.hpp"


#define MY_MIN(a, b) ((a) > (b) ? (b): (a))

unsigned char convert_value(float f) {
    int v = f * 255.0;
    if(v > 255) return 255;
    if(v < 0) return 0;
    return v;
}

class VideoMaker_BinaryRGB : public VideoMaker {
public:
    std::string filename;
    int w, h;
    int frame_index;

    VideoMaker_BinaryRGB(const std::string& filename_, int width, int height) {
        w = width; h = height;
        frame_index = 0;
        filename = filename_;
    }
    void addFrame(Vector4* rgba_array, int fw, int fh) {
        printf("Write frame: %d %d\n", fw, fh);
        bitmap_image image(fw, fh);
        for(int i = 0; i < fw * fh; i++) {
            int x = i % fw;
            int y = fh - i / fw - 1;
            image.set_pixel(x, y, convert_value(rgba_array[i].x), convert_value(rgba_array[i].y), convert_value(rgba_array[i].z));
        }
        char fn[256];
        sprintf(fn, ".%04d.bmp", frame_index++);
        image.save_image((filename + fn).c_str());
    }
    void finish() { }
    virtual ~VideoMaker_BinaryRGB() {
    }
};

VideoMaker::~VideoMaker() { }

VideoMaker* VideoMaker_Create(const std::string& filename, int width, int height) {
    return new VideoMaker_BinaryRGB(filename, width, height);
}
