#include "gui.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <unistd.h>
#include <X11/Xlib.h>

namespace pp {
namespace hw2 {
namespace nbody {

GUI::GUI(unsigned win_len, double coord_len, double x_min, double y_min) {
	// Set the width and the height
	win_len_ = win_len;
	scale_ = ((double) win_len_) / coord_len;
	x_min_ = x_min;
	y_min_ = y_min;

	// Open a connection to the x-window server
	display_ = XOpenDisplay(NULL);
	if(display_ == NULL) {
		fprintf(stderr, "cannot open display\n");
		exit(1);
	}

	// Get the id of the default screen
	screen_id_ = DefaultScreen(display_);

	// Generate the color pixels we need
	color_black_ = BlackPixel(display_, screen_id_);
	color_white_ = WhitePixel(display_, screen_id_);

	// Create a window
	window_ = XCreateSimpleWindow(display_, RootWindow(display_, screen_id_), 0, 0, win_len_, win_len_, 0, color_black_, color_white_);

	// Create a graphic context
	XGCValues values;
	gc_ = XCreateGC(display_, window_, 0, &values);
	//XSetForeground(display, gc, color_black_);
	//XSetBackground(display, gc, 0X0000FF00);
	//XSetLineAttributes(display, gc, 1, LineSolid, CapRound, JoinRound);

	// map(show) the window
	XMapWindow(display_, window_);
	XSync(display_, 0);
}

GUI::~GUI() {
}

void GUI::CleanAll() {
	// Draw a rectangle to clean everything
	XSetForeground(display_, gc_, color_black_);
	XFillRectangle(display_, window_, gc_, 0, 0, win_len_, win_len_);
}

void GUI::DrawAPoint(double coord_x, double coord_y) {
	XSetForeground(display_, gc_, color_white_);
	XDrawPoint(display_, window_, gc_, (unsigned) ((coord_x - x_min_) * scale_), (unsigned)((coord_y - y_min_) * scale_));
}

void GUI::Flush() {
	XFlush(display_);
}

} // namespace nbody
} // namespace hw2
} // namespace pp
