# Lighthouse Labs - Final Project
## Selective Object Recognition on Sight

This purpose of this application is to query information about an object in real-time based on a user's eye movement (i.e., what the user sees). This problem can be parted into 2 sections requiring use of different machine learning algorithms in each--1 regressor and 1 classifier. Due to time constraints, the scope of this project will be limited to a small but important subdivision of the eye-tracking component.

To better illustrate the idea, below is a birds-eye view of a possible solution subdivided into its key milestones:
1. eye-tracking
  - positional recognition of the user's pupil, approximated to a cartesian plane
  - unit vector interpretation of the user's gaze
2. object recognition in 3D space and classification output to markerless AR
  - image classifier requiring enormous amounts of labeled data about everyday objects (needs to be contextually limited to even start, otherwise the project may get stuck on trying to identify a fez hat from a mirror-glazed red velvet cake)
  - positional recognition of object edges to highlight and tag the object in AR space

This application will focus on applying regression models to existing eye image databases to determine the pupil's xy-coordinates.

### Notes

Srivastava, N., Newn, J., & Velloso, E. (2018). Combining Low and Mid-Level Gaze Features for Desktop Activity Recognition. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 2(4), 189.
