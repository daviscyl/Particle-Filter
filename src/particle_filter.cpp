/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::cos;
using std::normal_distribution;
using std::sin;
using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 25;  // Set the number of particles

  // Create normal (Gaussian) distributions for x, y, theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {
    Particle p{.id = i,
               .x = dist_x(gen),
               .y = dist_y(gen),
               .theta = dist_theta(gen),
               .weight = 1};

    particles.push_back(p);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  for (Particle& p : particles) {
    double new_theta = p.theta + yaw_rate * delta_t;
    double k = velocity / (yaw_rate + 1E-20);  // +1E-20 to avoid divide by zero

    // Move particle according to motion model
    p.x += k * (sin(new_theta) - sin(p.theta));
    p.y += k * (cos(p.theta) - cos(new_theta));
    p.theta = new_theta;

    // Create normal distributions for x, y, theta
    normal_distribution<double> dist_x(p.x, std_pos[0]);
    normal_distribution<double> dist_y(p.y, std_pos[1]);
    normal_distribution<double> dist_theta(p.theta, std_pos[2]);

    // Add noise to particle
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  for (auto& o : observations) {
    double min_dist = 1E63;

    for (int i = 0; i < predicted.size(); ++i) {
      LandmarkObs p = predicted[i];
      double d = dist(p.x, p.y, o.x, o.y);
      if (d < min_dist) {
        min_dist = d;
        o.id = i;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs>& observations,
                                   const Map& map_landmarks) {
  weights.clear();
  double weight_sum = 0;

  for (Particle& p : particles) {
    // 1. transform the observations into the map's coordinates
    vector<LandmarkObs> transformed_obs;
    p.associations.clear();
    p.sense_x.clear();
    p.sense_y.clear();

    double sinT = sin(p.theta);
    double cosT = cos(p.theta);

    for (auto obs : observations) {
      LandmarkObs obs_;
      obs_.x = p.x + cosT * obs.x - sinT * obs.y;
      obs_.y = p.y + sinT * obs.x + cosT * obs.y;
      transformed_obs.push_back(obs_);

      p.sense_x.push_back(obs_.x);
      p.sense_y.push_back(obs_.y);
    }

    // 2. find all landmards within sensor range to make the list of predicted
    // landmark observations
    vector<LandmarkObs> predicted_obs;

    for (auto lm : map_landmarks.landmark_list) {
      if (dist(p.x, p.y, lm.x_f, lm.y_f) < sensor_range) {
        LandmarkObs pred_obs{.id = lm.id_i, .x = lm.x_f, .y = lm.y_f};
        predicted_obs.push_back(pred_obs);
      }
    }

    // 3. find each landmark's closest observed point by modifying the "id"
    //    property of observation points in transformed_obs to
    //    the index of the closest point in predicted_obs. The "id" property of
    //    the observation points in predicted_obs corresponds with the actual
    //    map landmark's "id_i"
    dataAssociation(predicted_obs, transformed_obs);

    // 4. update the weights (particle & weights list) by multiplying the 2D
    // Gaussian probs
    p.weight = 1.0;
    for (auto obs : transformed_obs) {
      LandmarkObs pred = predicted_obs[obs.id];
      p.weight *= multiv_gauss(obs, pred, std_landmark);
      p.associations.push_back(pred.id);
    }

    weights.push_back(p.weight);
    weight_sum += p.weight;
  }

  // 5. normalize the weights to a sum of 1.0
  for (int i = 0; i < num_particles; ++i) {
    weights[i] /= weight_sum;
    particles[i].weight = weights[i];
  }
}

void ParticleFilter::resample() {
  vector<Particle> new_particles;
  std::discrete_distribution<> d(weights.begin(), weights.end());

  for (int i = 0; i < num_particles; ++i) {
    Particle old_p = particles[d(gen)];
    Particle new_p{.id = i,
                   .x = old_p.x,
                   .y = old_p.y,
                   .theta = old_p.theta,
                   .weight = old_p.weight,
                   .associations = old_p.associations,
                   .sense_x = old_p.sense_x,
                   .sense_y = old_p.sense_y};
    new_particles.push_back(new_p);
  }

  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}