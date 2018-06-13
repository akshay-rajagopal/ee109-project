# Milestone 2 Submission

## Hardware Implementation
```scala
// Please attach your Spatial implementation here.
```
// Digits SVM
object ProjectSVM extends SpatialApp {

  @struct class SVMResult(
    margin: Float,
    digit: Int
  )

  @virtualize
  def main() {

    val outerPar = 1
    val midPar = 2
    val innerPar = 2

    //type T = FixPt[TRUE,_4,_12]
    type T = Float
    val tileM = 16
    val tileN = 16
    val tileK = 16

    val picSize = 400.to[Int]
    val numTrainImages = 60000.to[Int] // 600 for simulation
    val numTestImages = 10000.to[Int] // 100 for simulation
    val digits = 10.to[Int]

    val rho = 0.04.to[T]
    val base_alpha = 0.0001.to[T]
    val success_alpha = 0.0005.to[T]

    // For testing, I used the '_med' files
    val train_data = loadCSV2D[T]("/home/akshayr2/ee109-project/ProjectHardware/images_train_med.csv",",")
    val train_labels = loadCSV2D[Int]("/home/akshayr2/ee109-project/ProjectHardware/labels_train_med.csv",",")
    val test_data = loadCSV2D[T]("/home/akshayr2/ee109-project/ProjectHardware/images_test_med.csv",",")
    val test_labels = loadCSV2D[Int]("/home/akshayr2/ee109-project/ProjectHardware/labels_test_med.csv",",")
  
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
            mux(W_sram(i,j) == 0.to[T] || img_sram(0,j) == 0.to[T], 0.to[T],mux(y > 0, W_sram(i,j) * img_sram(0,j), -W_sram(i,j) * img_sram(0,j)))
          }{_ + _}
          val select = 1.to[T] - ywx.value
          val gk1_sram = SRAM[T](picSize)
          Foreach(picSize by 1){j =>
	    val rhoW = mux(W_sram(i,j) == 0.to[T], 0.to[T], rho * W_sram(i,j))
            gk1_sram(j) = mux(select > 0 && img_sram(0,j) != 0.to[T], rhoW - y.to[T] * img_sram(0,j),rhoW)
          }
          Foreach(picSize by 1){j =>
            W_sram(i,j) = mux(gk1_sram(j) == 0.to[T], W_sram(i,j) ,W_sram(i,j) - alpha * gk1_sram(j))
          }
        }
      }
      W(0::digits,0::picSize) store W_sram

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
          mux(W_sram(0,j) == 0.to[T] || img_sram(0,j) == 0.to[T], 0.to[T], W_sram(0,j) * img_sram(0,j))
        }{_ + _}
	

        Sequential.Foreach(1 until digits by 1){ i =>
          val res = Reg[T](0)
          Reduce(res)(picSize by 1){j =>
            mux(W_sram(i,j) == 0.to[T] || img_sram(0,j) == 0.to[T], 0.to[T], W_sram(i,j) * img_sram(0,j))
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

## Scala Simulation Result
```bash
# Please attach the Scala functional simulation result here.
```
Errors: 42
[success] Total time: 768 s, completed May 23, 2018 10:12:05 PM

By contrast, MATLAB on the same set had 34 errors.  This result was obtained after training on 600 images and testing on 100.
Our hardware SVM is indeed learning, and the difference is likely due to floating point differences.  

## VCS Simulation Result
```bash
# Please attach the VCS simulation result here.
```
We were told during the meeting on Monday that the VCS simulation was not actually necessary for today.
