#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	// taken from Udacity EKF solutions
	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;


	// check the validity of the following inputs:
	//  * the estimation vector size should not be zero
	//  * the estimation vector size should equal ground truth vector size
	if (estimations.size() == 0 || estimations.size() != ground_truth.size()) {
		cout << "Error: Invalid Data" << endl;
		return rmse;
	}

	VectorXd residuals;
	//accumulate squared residuals
	for (int i = 0; i < estimations.size(); ++i) {
		residuals = estimations[i] - ground_truth[i];
		residuals = residuals.array()*residuals.array();
		rmse += residuals;
	}
	//calculate the mean
	rmse /= estimations.size();
	//calculate the squared root
	rmse = rmse.array().sqrt();

	//return the result
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
	// taken from Udacity EKF lesson solutions
	MatrixXd Hj(3, 4);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//check division by zero
	if (px == 0 && py == 0) {
		cout << "Division by zero error" << endl;
		return Hj;
	}
	float c1 = px * px + py * py;
	float c2 = sqrt(px*px + py * py);
	float c3 = c1 * c2;
	//compute the Jacobian matrix
	Hj << px / c2, py / c2, 0, 0,
		-py / c1, px / c1, 0, 0,
		py*(vx*py - vy * px) / c3, px*(vy*px - vx * py) / c3, px / c2, py / c2;
	return Hj;
}
