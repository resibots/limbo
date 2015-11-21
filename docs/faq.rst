Frequently Asked Questions
==========================

Why using waf (and not Cmake, or <insert your favorite build system>)?
--------------------------------------------------------------------------

Short answer: because we used it in `Sferes2 <http://www.github.com/sferes2/sferes>`_ and we liked it!

Here is the longer answer. When we started Sferes2, around 2007, the free software community was starting to be tired of Automake/Autoconf (the *de facto* standard at that time) and it was looking for more modern alternatives. Two/Three software had some tractions:

- `Cmake <http://www.cmake.org>`_, which generates makefiles (like Autoconf / Automake), uses a custom language, and emphasizes Windows/Unix portability;

- `Scons <http://www.scons.org>`_, which can be seen as a set of high level python functions / objects to build software;

- `Waf <http://www.waf.org>`_, which was a bit like "a simpler scons"

We tried all of them.

Cmake uses a custom language that is not especially well designed in our opinion (it looks like a language from the 70s and it ignores most of the research in computer languages of the last 30 years). Generating makefiles is nice, but (1) it is slow (because generating files is slow), (2) it is a 40 year old syntax (which does the job, OK), and (3) it is not designed for parallel build (whereas all our computers are now multi-core). Compiling on both Unix and MS Windows is easy, but Windows is not our target platform. On the contrary, cross-compiling for embedded platforms (e.g. am ARM-based Raspberry Pi or a robot) was very hard with the first version of Cmake (whereas it was easy with Autoconf / Automake !).

Scons is nice because it is based on Python, a nice, general purpose language that almost everybody know. It makes it easy to build complex build frameworks, but simple things can be not-that-simple to do (at least, in 2007, it probably improved since then). In addition, Scons does not generate makefiles, which makes it faster and more adapted to parallel builds.

Waf has all the advantages of Scons but the learning curve is less steep. It is based on Python, it has a nice modern feel, it is fast, and it parallelizes the builds by default. Using python also allows us to easily add services that is not typically part of a build system, for instance, to submit jobs to a cluster (we have to parse a JSON file, etc., which is trivial in python, but not in Cmake!).

Overall, it seems that cmake won the "war" in the free software world, mainly because a few high-profile projects chose it instead of Scons or Waf -- the most prominent (and "contagious") project is most probably QT. However, the battle was tough. For instance, I remember that no build system was perfect for QT and someone even made a waf-based version of QT. Also, keep in mind that QT needs a build system that works very well on MS Windows, while we do not care (we have no Windows-based clusters and no Windows-based robots...): they had different goals. In robotics, ROS heavily relies on Cmake, but one could wonder if Catkin would have been faster/nicer/better if it had been based on waf or scons. Last, a few high profile projects chose waf. For instance, `Pebble <http://www.pebble.com>`_, the smart watches, or `Samba <http://www.samba.org>`_, the Windows-compatible file sharing system for Unix.

Wny do you choose to not use configuration files?
--------------------------------------------------

Short answer: because we target developpers/researchers who want to write the smallest amount of code when they add a new functionnality/concept, and not "end-users" who want an external optimizer that they can easily call on their problem.

Long answer is :ref:`here <params-guide>`.

Why C++? (and not <insert your favorite language>)?
---------------------------------------------------
