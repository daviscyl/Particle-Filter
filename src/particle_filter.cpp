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
  /**
   * TODO: √ Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: √ Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  num_particles = 100;  // FIXME: Set the number of particles

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
  /**
   * TODO: √ Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  for (Particle& p : particles) {
    float new_theta = p.theta + yaw_rate * delta_t;
    float k = velocity / yaw_rate;

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
  /**
   * TODO: √ Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */
  for (auto& o : observations) {
    LandmarkObs* closest = NULL;

    for (int i = 0; i < predicted.size(); ++i) {
      LandmarkObs p = predicted[i];
      if (closest == NULL ||
          dist(p.x, p.y, o.x, o.y) < dist(closest->x, closest->y, o.x, o.y)) {
        closest = &p;
        o.id = i;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs>& observations,
                                   const Map& map_landmarks) {
  /**
   * TODO: √ Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no
   *   scaling). The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  weights.clear();
  double weight_sum = 0;

  for (Particle& p : particles) {
    // 1. transform the observations into the map's coordinates
    vector<LandmarkObs> transformed_observations;
    p.associations.clear();
    p.sense_x.clear();
    p.sense_y.clear();

    double sinT = sin(p.theta);
    double cosT = cos(p.theta);

    for (auto obs : observations) {
      LandmarkObs trans_obs;
      trans_obs.x = p.x + cosT * obs.x - sinT * obs.y;
      trans_obs.y = p.y + sinT * obs.x + cosT * obs.y;
      transformed_observations.push_back(trans_obs);

      p.sense_x.push_back(trans_obs.x);
      p.sense_y.push_back(trans_obs.y);
    }

    // 2. find all landmards within sensor range to make the list of predicted
    // landmark observations
    vector<LandmarkObs> predicted_observatons;

    for (auto lm : map_landmarks.landmark_list) {
      if (dist(p.x, p.y, lm.x_f, lm.y_f) < sensor_range) {
        LandmarkObs pred_obs{.id = lm.id_i, .x = lm.x_f, .y = lm.y_f};
        predicted_observatons.push_back(pred_obs);
      }
    }

    // 3. find each landmark's closest observed point
    dataAssociation(predicted_observatons, transformed_observations);

    // 4. update the weights (particle & weights list) by multiplying the 2D
    // Gaussian probs
    p.weight = 1.0;
    for (auto obs : transformed_observations) {
      p.weight *= multiv_gauss(obs, predicted_observatons[obs.id], std_landmark);
      p.associations.push_back(predicted_observatons[obs.id].id);
    }

    weights.push_back(p.weight);
    weight_sum += p.weight;
  }
  // std::cout << "weight sum: " << weight_sum << std::endl;

  for (int i = 0; i < num_particles; ++i) {
    // std::cout << "before normalization: " << std::endl
    // << "weights list: " << weights[i] << "\t"
    // << "particle weight: " << particles[i].weight << std::endl;
    weights[i] /= weight_sum;
    particles[i].weight /= weight_sum;
    // std::cout << "after normalization: " << std::endl
    // << "weights list: " << weights[i] << "\t"
    // << "particle weight: " << particles[i].weight << std::endl;
  }

  weight_sum = 0;
  for (auto w : weights) {
    weight_sum += w;
  }

  // std::cout << "new weight sum: " << weight_sum << std::endl;
}

void ParticleFilter::resample() {
  /**
   * TODO: √ Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> new_particles;
  std::discrete_distribution<> d(weights.begin(), weights.end());

  for (int i = 0; i < num_particles; ++i) {
    Particle p_old = particles[d(gen)];
    Particle p_new{.id = i, .x = p_old.x, .y = p_old.y, .theta = p_old.theta};
    new_particles.push_back(p_new);
  }

  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
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