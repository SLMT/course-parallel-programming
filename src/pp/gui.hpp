#ifndef PP_GUI_H_
#define PP_GUI_H_

#include <X11/Xlib.h>

namespace pp {

class GUI {

public:
	GUI(unsigned win_width, unsigned win_height);
	GUI(unsigned win_len, double coord_len, double x_min, double y_min);
	~GUI();

	void CleanAll();
	void DrawAPoint(double x, double y);
	void DrawAPoint(double x, double y, unsigned long color);
	void DrawALine(double start_x, double start_y, double end_x, double end_y);
	void Flush();

private:
	void InitXWindow();

	inline unsigned MapX(double coord_x) {
		return (unsigned) ((coord_x - x_min_) * x_scale_);
	}

	inline unsigned MapY(double coord_y) {
		return (unsigned) ((coord_y - y_min_) * y_scale_);
	}

	// Some attributes
	unsigned win_width_, win_height_;
	double x_scale_, y_scale_;
	double x_min_, y_min_;

	// X-Window Componenets
	Display *display_;
	Window window_;
	GC gc_;
	int screen_id_;

	static const unsigned long kColorWhite = 0xFFFFFFL, kColorBlack = 0x000000L, kColorGray = 0x5F5F5FL;
};

} // namespace pp


#endif  // PP_GUI_H_
