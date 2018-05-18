import DataImplicits._

abstract class Counterlike {
  def foreach(func: (Array[FixedPoint],Array[Bool]) => Unit): Unit
}

case class Counter(start: FixedPoint, end: FixedPoint, step: FixedPoint, par: FixedPoint) extends Counterlike {
  private val parStep = par
  private val fullStep = parStep * step
  private val vecOffsets = Array.tabulate(par){p => p * step}

  def foreach(func: (Array[FixedPoint],Array[Bool]) => Unit): Unit = {
    var i = start
    while (if (step > 0) {i < end} else {i > end}) {
      val vec = vecOffsets.map{ofs => FixedPoint(ofs + i) } // Create current vector
      val valids = vec.map{ix => Bool(if (step > 0) {ix < end} else {ix > end}) }        // Valid bits
      func(vec, valids)
      i += fullStep
    }
  }
}

case class Forever() extends Counterlike {
  def foreach(func: (Array[FixedPoint],Array[Bool]) => Unit): Unit = {
    val vec = Array.tabulate(1){ofs => FixedPoint(ofs) }  // Create current vector
    val valids = vec.map{ix => Bool(true) }               // Valid bits

    while (true) {
      func(vec, valids)
    }
  }
}