import AssemblyKeys._

name := "spark-ml"

scalaVersion := "2.10.3"

version := "1.2"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.2.0-SNAPSHOT"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.2.0-SNAPSHOT"

libraryDependencies += "org.apache.spark" %% "spark-sql" % "1.2.0-SNAPSHOT"

libraryDependencies += "org.scalatest" %% "scalatest" % "1.9.1" % "test"

assemblySettings

assemblyOption in assembly ~= { _.copy(includeScala = false) }

test in assembly := {}

mergeStrategy in assembly <<= (mergeStrategy in assembly) { (old) =>
  {
    case PathList("javax", "servlet", xs @ _*)           => MergeStrategy.first
    case PathList(ps @ _*) if ps.last endsWith ".html"   => MergeStrategy.first
    case "application.conf"                              => MergeStrategy.concat
    case "reference.conf"                                => MergeStrategy.concat
    case "log4j.properties"                              => MergeStrategy.discard
    case m if m.toLowerCase.endsWith("manifest.mf")      => MergeStrategy.discard
    case m if m.toLowerCase.matches("meta-inf.*\\.sf$")  => MergeStrategy.discard
    case _ => MergeStrategy.first
  }
}
