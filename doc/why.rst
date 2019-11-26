
Why joblib: project goals
=========================

Joblibspark's approach
-----------------

Functions are the simplest abstraction used by everyone. Pipeline
jobs (or tasks) in Joblib are made of decorated functions.

Tracking of parameters in a meaningful way requires specification of
data model. Joblib gives up on that and uses hashing for performance and
robustness.

Design choices
--------------

* No dependencies other than Python

* Robust, well-tested code, at the cost of functionality

* Fast and suitable for scientific computing on big dataset without
  changing the original code

* Only local imports: **embed joblib in your code by copying it**



