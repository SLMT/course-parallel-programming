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

	// Create a window
	window_ = XCreateSimpleWindow(display_, RootWindow(display_, screen_id_), 0, 0, win_len_, win_len_, 0, kColorBlack, kColorWhite);

	// Create a graphic context
	XGCValues values;
	gc_ = XCreateGC(display_, window_, 0, &values);

	// map(show) the window
	XMapWindow(display_, window_);
	XSync(display_, 0);
}

GUI::~GUI() {
}

void GUI::CleanAll() {
	// Draw a rectangle to clean everything
	XSetForeground(display_, gc_, kColorBlack);
	XFillRectangle(display_, window_, gc_, 0, 0, win_len_, win_len_);
}

void GUI::DrawAPoint(double x, double y) {
	XSetForeground(display_, gc_, kColorWhite);
	XDrawPoint(display_, window_, gc_, MapX(x), MapY(y));
}

void GUI::DrawALine(double start_x, double start_y, double end_x, double end_y) {
	XSetForeground(display_, gc_, kColorGray);
	XDrawLine(display_, window_, gc_, MapX(start_x), MapY(start_y), MapX(end_x), MapY(end_y));
}

void GUI::Flush() {
	XFlush(display_);
}

} // namespace nbody
} // namespace hw2
} // namespace pp
