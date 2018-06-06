import spatial.dsl._
import org.virtualized._

// Digits SVM
object ProjectSVM extends SpatialApp {

  @struct class SVMResult(
    margin: FixPt[TRUE,_8,_24],
    digit: Int
  )

  @virtualize
  def main() {

    val outerPar = 1
    val midPar = 2
    val innerPar = 32

    type T = FixPt[TRUE,_5,_27]
    //type T = Float
    val tileM = 16
    val tileN = 16
    val tileK = 16

    val picSize = 400.to[Int]
    val numTrainImages = 60000.to[Int] // Real value
    val numTestImages = 10000.to[Int] // Real value
    //val numTrainImages = 600.to[Int] // For simulation
    //val numTestImages = 100.to[Int] // For simulation
    val digits = 10.to[Int]

    val rho = 0.04.to[T]
    val base_alpha = 0.0001.to[T]
    val success_alpha = 0.0005.to[T]

    // For testing, I used the '_med' files
    val train_data = loadCSV2D[T]("images_train.csv",",")
    val train_labels = loadCSV2D[Int]("labels_train.csv",",")
    val test_data = loadCSV2D[T]("images_test.csv",",")
    val test_labels = loadCSV2D[Int]("labels_test.csv",",")
  
    val W_init = (0::digits, 0::picSize){(i,j) => 0.to[T]}
    val trainImages = DRAM[T](numTrainImages, picSize)
    val trainLabels = DRAM[Int](numTrainImages)
    val testImages = DRAM[T](numTestImages, picSize)
    val testLabels = DRAM[Int](numTestImages)
    val W = DRAM[T](digits, picSize) 
 
    println("Done Loading")
    setMem(trainImages, train_data)
    setMem(trainLabels, train_labels)
    setMem(testImages, test_data)
    setMem(testLabels, test_labels)
    setMem(W, W_init)

    val argErrorsOut = ArgOut[Int]
    Accel {
      val trainlabels_sram = SRAM[Int](numTrainImages)
      trainlabels_sram load trainLabels(0::numTrainImages)
      val W_sram = SRAM[T](digits, picSize)
      W_sram load W(0::digits, 0::picSize)
      Sequential.Foreach(numTrainImages by 1){k =>
        val img_sram = SRAM[T](1,picSize)
        img_sram load trainImages(k::k+1, 0::picSize)
        val label = Reg[Int](0)
        label := trainlabels_sram(k)
        Sequential.Foreach(digits by 1) {i =>
          val y = mux(i.to[Int] == label.value, 1.to[Int], -1.to[Int])
          //val alpha = mux(i.to[Int] == label, 10/k, 2/k)
          val alpha = mux(i.to[Int] == label.value, success_alpha, base_alpha)
          val ywx = Reg[T](0)
          Reduce(ywx)(picSize by 1 par innerPar) {j =>
            y.to[T] * W_sram(i,j) * img_sram(0,j)
          }{_ + _}
          val select = 1.to[T] - ywx.value
          val gk1_sram = SRAM[T](picSize)
          Foreach(picSize by 1 par innerPar){j =>
            gk1_sram(j) = mux(select > 0 , rho * W_sram(i,j) - y.to[T] * img_sram(0,j), rho * W_sram(i,j))
          }
          Foreach(picSize by 1 par innerPar){j =>
            W_sram(i,j) = W_sram(i,j) - alpha * gk1_sram(j)
          }
        }
      }
      W(0::digits,0::picSize) store W_sram

      val errors = Reg[Int](0)
      val testlabels_sram = SRAM[Int](numTestImages)
      testlabels_sram load testLabels(0::numTestImages)
            
      Sequential.Foreach(numTestImages by 1){ k =>

        val img_sram = SRAM[T](1, picSize)
        val label = Reg[Int](0)
        val maxvalinit = Reg[T](0)
        img_sram load testImages(k::k+1, 0::picSize)
        label := testlabels_sram(k)

        val probs = SRAM[T](digits)
        Foreach(0 until digits by 1 par digits){i =>
          val res = Reg[T](0)
          Reduce(res)(picSize by 1 par innerPar){j =>
            W_sram(i,j) * img_sram(0,j)
          }{_+_}
          probs(i) = res.value
        }

        val maxind = Reg[Int](0)
        val maxval = Reg[T](0)

        Sequential.Foreach(0 until digits by 1){ i =>
          if (i == 0) {
            // init
            maxval := probs(0)
            maxind := 0.to[Int]
          } else {
            // otherwise
            val oldval = maxval.value
            maxval := mux(probs(i) >= oldval, probs(i), oldval)
            maxind := mux(probs(i) >= oldval, i.to[Int], maxind.value)
          }
        }
        errors := mux(maxind.value == label.value, errors.value, errors.value + 1)
      }
      argErrorsOut := errors.value
    }

    val accel_matrix = getMatrix(W)
    val errorsResult = getArg(argErrorsOut)
    println("Errors: " + errorsResult)
  }
}

object ProjectNN extends SpatialApp {

  @virtualize
  def main() {

    val outerPar = 1
    val midPar = 16
    val innerPar = 16

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