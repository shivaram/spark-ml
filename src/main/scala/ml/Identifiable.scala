package ml

import java.util.UUID

/**
 * Something with a unique id.
 */
abstract class Identifiable {
  var id: String = getClass.getName + Identifiable.randomId() 

  def setId(idStr: String) = {
    id = idStr
  }
}

object Identifiable {
  private[ml] def randomId(): String = UUID.randomUUID().toString.take(8)
}
