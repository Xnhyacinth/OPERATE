"""Logistics-domain seed factories.

Build ``LogisticsScenarioSeed`` objects from real public VRP datasets:

- ``from_vrplib``       — Augerat / Uchoa CVRP and Solomon / Gehring-Homberger
                          VRPTW instances parsed from the MIT-mirrored
                          ``PyVRP/Instances`` set anchored under ``works/``.
- ``from_amazon_lmrrc`` — Amazon Last-Mile Routing Research Challenge
                          (CC-BY-NC-4.0) stops / priorities / zones for the
                          last-mile priority family.

All released scenarios derive from a manifest-declared real dataset with
``provenance`` URL / commit / license / file (Red Line #2). The integer
``seed`` is *structural* (§5 of the spec), not fog-only.
"""
