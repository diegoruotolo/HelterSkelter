// Sonatype publishing (Maven Central via central.sonatype.com)
addSbtPlugin("org.xerial.sbt" % "sbt-sonatype" % "3.12.2")

// GPG signing — required by Maven Central
addSbtPlugin("com.github.sbt" % "sbt-pgp" % "2.2.1")

// Automated version bump + Git tagging
addSbtPlugin("com.github.sbt" % "sbt-release" % "1.4.0")

