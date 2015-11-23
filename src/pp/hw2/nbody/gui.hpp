#ifndef PP_HW2_NBODY_GUI_H_
#define PP_HW2_NBODY_GUI_H_

#include <X11/Xlib.h>

namespace pp {
namespace hw2 {
namespace nbody {

class GUI {

public:
	GUI(unsigned width, unsigned height);
	~GUI();

	void CleanAll();
	void DrawAPoint(unsigned x, unsigned y);
	void Flush();

private:
	// Some attributes
	unsigned width_, height_;

	// X-Window Componenets
	Display *display_;
	Window window_;
	GC gc_;
	int screen_id_;
	unsigned long color_white_, color_black_;
};

} // namespace nbody
} // namespace hw2
} // namespace pp


#endif  // PP_HW2_NBODY_GUI_H_
