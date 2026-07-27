"""
Authors: Wade Williams, Thomas Kulch, Darshan Kedari

Purpose: Orchestration -- run one Frame end to end and return what a caller needs to visualize
or score it. The only module that imports every other; nothing may import it back.

Build this last, once the modules it composes exist. Construct expensive objects (the detector,
the depth model) ONCE and reuse them across frames rather than rebuilding per call.
"""
