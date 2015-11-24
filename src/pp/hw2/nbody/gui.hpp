#ifndef PP_HW2_NBODY_GUI_H_
#define PP_HW2_NBODY_GUI_H_

#include <X11/Xlib.h>

namespace pp {
namespace hw2 {
namespace nbody {

class GUI {

public:
	GUI(unsigned win_len, double coord_len, double x_min, double y_min);
	~GUI();

	void CleanAll();
	void DrawAPoint(double coord_x, double coord_y);
	void Flush();

private:
	// Some attributes
	unsigned win_len_;
	double scale_;
	double x_min_, y_min_;

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
