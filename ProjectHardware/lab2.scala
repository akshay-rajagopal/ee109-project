import spatial.dsl._
import org.virtualized._


// GEMM
object Lab2Part5GEMM extends SpatialApp {

  @virtualize
  def main() {

    val outerPar = 1
    val midPar = 2
    val innerPar = 2

    type T = FixPt[TRUE,_24,_8]
    val tileM = 16
    val tileN = 16
    val tileK = 16

    val M = ArgIn[Int]
    val N = ArgIn[Int]
    val K = ArgIn[Int]
    setArg(M,args(0).to[Int])
    setArg(N,args(1).to[Int])
    setArg(K,args(2).to[Int])

    val a_data = (0::args(0).to[Int], 0::args(2).to[Int]){(i,j) => random[T](3)}
    val b_data = (0::args(2).to[Int], 0::args(1).to[Int]){(i,j) => random[T](3)}
    val c_init = (0::args(0).to[Int], 0::args(1).to[Int]){(i,j) => 0.to[T]}
    val a = DRAM[T](M, K)
    val b = DRAM[T](K, N)
    val c = DRAM[T](M, N)

    setMem(a, a_data)
    setMem(b, b_data)
    setMem(c, c_init)

    Accel {
      Foreach(K by tileK par outerPar){kk =>
        val numel_k = min(tileK.to[Int], K - kk)
        Foreach(M by tileM par innerPar){mm =>
          val numel_m = min(tileM.to[Int], M - mm)
          val tileA_sram = SRAM[T](tileM, tileK)
          tileA_sram load a(mm::mm+numel_m, kk::kk+numel_k)
          Foreach(N by tileN par innerPar){nn =>
            val numel_n = min(tileN.to[Int], N - nn)
            val tileB_sram = SRAM[T](tileK, tileN)
            val tileC_sram = SRAM.buffer[T](tileM, tileN)
            tileB_sram load b(kk::kk+numel_k, nn::nn+numel_n)
            tileC_sram load c(mm::mm+numel_m, nn::nn+numel_n)

            // Your code here
            MemFold(tileC_sram)(0 until numel_k by 1){k =>
              val tmp = SRAM[T](tileM,tileN)
              Foreach(0 until numel_m by 1 par tileM) { i => 
                Foreach(0 until numel_n by 1 par tileN) {j => 
                  tmp(i,j) = tileA_sram(i,k)*tileB_sram(k,j)
                }
              }
              tmp
            }{_+_}
            c(mm::mm+numel_m, nn::nn+numel_n) store tileC_sram
          }
        }
      }
    }

    val accel_matrix = getMatrix(c)
    val gold_matrix = (0::args(0).to[Int], 0::args(1).to[Int]){(i,j) =>
      Array.tabulate(args(2).to[Int]){k => a_data(i,k) * b_data(k,j)}.reduce{_+_}
    }

    printMatrix(accel_matrix, "Received: ")
    printMatrix(gold_matrix, "Wanted: ")
    val cksum = accel_matrix.zip(gold_matrix){_==_}.reduce{_&&_}
    println("PASS: " + cksum + "(Lab2Part5GEMM)")
  }
}

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
    val innerPar = 2

    type T = FixPt[TRUE,_8,_24]
    //type T = Float
    val tileM = 16
    val tileN = 16
    val tileK = 16

    val picSize = 400.to[Int]
    //val numTrainImages = 60000.to[Int] // 600 for simulation
    //val numTestImages = 10000.to[Int] // 100 for simulation
    val numTrainImages = 600.to[Int] // 600 for simulation
    val numTestImages = 100.to[Int] // 100 for simulation
    val digits = 10.to[Int]

    val rho = 0.04.to[T]
    val base_alpha = 0.001.to[T]
    val success_alpha = 0.005.to[T]

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
      Foreach(numTrainImages by 1){k =>
        val img_sram = SRAM[T](1,picSize)
        img_sram load trainImages(k::k+1, 0::picSize)
        val label = Reg[Int](0)
        label := trainlabels_sram(k)
        Foreach(digits by 1) {i =>
          val y = mux(i.to[Int] == label.value, 1.to[Int], -1.to[Int])
          //val alpha = mux(i.to[Int] == label, 10/k, 2/k)
          val alpha = mux(i.to[Int] == label.value, success_alpha, base_alpha)
          val ywx = Reg[T](0)
          Reduce(ywx)(picSize by 1) {j =>
            y.to[T] * W_sram(i,j) * img_sram(0,j)
          }{_ + _}
          val select = 1.to[T] - ywx.value
          val gk1_sram = SRAM[T](picSize)
          Foreach(picSize by 1){j =>
            gk1_sram(j) = mux(select > 0 , rho * W_sram(i,j) - y.to[T] * img_sram(0,j), rho * W_sram(i,j))
          }
          Foreach(picSize by 1){j =>
            W_sram(i,j) = W_sram(i,j) - alpha * gk1_sram(j)
          }
        }
      }
      W(0::digits,0::picSize) store W_sram
      Foreach(digits by 1){i=>println(W_sram(i,19))}

      val errors = Reg[Int](0)
      val testlabels_sram = SRAM[Int](numTestImages)
      testlabels_sram load testLabels(0::numTestImages)
      Foreach(numTestImages by 1){k =>
        val img_sram = SRAM[T](1, picSize)
        img_sram load testImages(k::k+1, 0::picSize)
        val label = Reg[Int](0)
        label := testlabels_sram(k)
        val maxind = Reg[Int](0)
        val maxval = Reg.buffer[T](0)
        Reduce(maxval)(picSize by 1){j =>
          W_sram(0,j) * img_sram(0,j)
        }{_ + _}
	

        Sequential.Foreach(1 until digits by 1){ i =>
          val res = Reg[T](0)
          Reduce(res)(picSize by 1){j =>
            W_sram(i,j) * img_sram(0,j)
          }{_ + _}
          maxval := mux(res.value >= maxval.value, res.value, maxval.value)
          maxind := mux(res.value == maxval.value, i.to[Int], maxind.value)
        }
        errors := mux(maxind.value == label.value, errors.value, errors.value + 1)
      }
      argErrorsOut := errors.value
    }

    val accel_matrix = getMatrix(W)
    //val gold_matrix = (0::args(0).to[Int], 0::args(1).to[Int]){(i,j) =>
    //  Array.tabulate(args(2).to[Int]){k => a_data(i,k) * b_data(k,j)}.reduce{_+_}
    //}
    val errorsResult = getArg(argErrorsOut)
    println("Errors: " + errorsResult)
    //printMatrix(accel_matrix, "Received: ")
    
    //printMatrix(gold_matrix, "Wanted: ")
    //val cksum = accel_matrix.zip(gold_matrix){_==_}.reduce{_&&_}
    //println("PASS: " + cksum + "(Lab2Part5GEMM)")
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
