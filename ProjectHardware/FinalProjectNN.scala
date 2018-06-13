import spatial.dsl._
import virtualized._

// Digits NN
object ProjectNNSynthZCU extends SpatialApp {

  @virtualize
  def main() {

    val outerPar = 1
    val midPar = 8  // Originally 16 on Arria10
    val innerPar = 8  // Originally 16 on Arria10

    type T = FixPt[TRUE,_16,_16]
    val tileM = 16
    val tileN = 16
    val tileK = 16

    val picSize = 784.to[Int]
    val nhidden_1 = 1024.to[Int]
    val numTestImages = 10000.to[Int] // 200 for simulation
    val digits = 10.to[Int]

    val W1_data = loadCSV2D[T]("W1.csv",",")
    val b1_data = loadCSV1D[T]("b1.csv",",")
    val W2_data = loadCSV2D[T]("W2.csv",",")
    val b2_data = loadCSV1D[T]("b2.csv",",")
    val test_data = loadCSV2D[T]("mnist_test_images_28.csv",",")
    val test_labels = loadCSV1D[Int]("mnist_test_labels_28.csv",",")

    val testImages = DRAM[T](numTestImages, picSize)
    val testLabels = DRAM[Int](numTestImages)
    val W1 = DRAM[T](nhidden_1, picSize)
    val b1 = DRAM[T](nhidden_1)
    val W2 = DRAM[T](digits, nhidden_1)
    val b2 = DRAM[T](digits)

    println("Done Loading")
    setMem(testImages, test_data)
    setMem(testLabels, test_labels)
    setMem(W1, W1_data)
    setMem(b1, b1_data)
    setMem(W2, W2_data)
    setMem(b2, b2_data)


    val argErrorsOut = ArgOut[Int]
    Accel {
      val b1_sram = SRAM[T](nhidden_1)
      b1_sram load b1(0::nhidden_1)
      val b2_sram = SRAM[T](digits)
      b2_sram load b2(0::digits)
      val W2_sram = SRAM[T](digits, nhidden_1)
      W2_sram load W2(0::digits, 0::nhidden_1)

      val errors = Reg[Int](0)
      val testlabels_sram = SRAM[Int](numTestImages)
      testlabels_sram load testLabels(0::numTestImages)
      Sequential.Foreach(numTestImages by 1){k =>
        val img_sram = SRAM[T](1, picSize)
        img_sram load testImages(k::k+1, 0::picSize)
        val label = Reg[Int](0)
        label := testlabels_sram(k)

        val inter = SRAM[T](nhidden_1)
        Foreach(nhidden_1 by 1 par midPar){ i =>
          val W1_neuron = SRAM[T](1,picSize)
          W1_neuron load W1(i::i+1,0::picSize)
          val res = Reg[T](0)
          Reduce(res)(picSize by 1 par innerPar){j =>
             W1_neuron(0,j) * img_sram(0,j)
          }{_ + _}
          inter(i) = max(res.value + b1_sram(i), 0.to[T])
        }

        val probs = SRAM[T](digits)
        Foreach(digits by 1 par digits){ i =>
          val res = Reg[T](0)
          Reduce(res)(nhidden_1 by 1 par innerPar){j =>
            W2_sram(i,j) * inter(j)
          }{_ + _}
          probs(i) = res.value + b2_sram(i)
        }

        val maxind = Reg.buffer[Int](0)
        val maxval = Reg.buffer[T](0)
        maxind := 0.to[Int]
        maxval := probs(0)

        Sequential.Foreach(1 until digits by 1){ i =>
          val oldval = maxval.value
          maxval := mux(probs(i) > oldval, probs(i), oldval)
          maxind := mux(probs(i) > oldval, i.to[Int], maxind.value)
        }
        println("Max val: " + maxval.value)
        println("Max ind: " + maxind.value)
        errors := mux(maxind.value == label.value, errors.value, errors.value + 1)
      }
      argErrorsOut := errors.value
    }
    val errorsResult = getArg(argErrorsOut)
    println("Errors: " + errorsResult)
  }
}