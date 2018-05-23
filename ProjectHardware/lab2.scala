import spatial.dsl._
import org.virtualized._


// MemReduce
object Lab2Part1SimpleMemReduce extends SpatialApp {

  val N = 16.to[Int]

  @virtualize
  def main() {
    val out = DRAM[Int](16)
    Accel {
      val a = SRAM[Int](16)
      MemReduce(a)(-5 until 5 by 1){i =>
        val tmp = SRAM[Int](16)
        Foreach(16 by 1) { j => tmp(j) = 1}
        tmp
      }{_+_}
      out store a
    }

    val result = getMem(out)
    val gold = Array.tabulate(16){i => 10.to[Int]}
    printArray(gold, "expected: ")
    printArray(result, "result:   ")

    val cksum = gold.zip(result){_==_}.reduce{_&&_}
    println("PASS: " + cksum + " (Lab2Part1SimpleMemReduce)")
  }
}


// MemFold
object Lab2Part2SimpleMemFold extends SpatialApp {

  val N = 16.to[Int]

  @virtualize
  def main() {
    val out = DRAM[Int](16)
    Accel {
      val a = SRAM[Int](16)
      Foreach(16 by 1) { j => a(j) = 0}
      MemFold(a)(-5 until 5 by 1){i =>
        val tmp = SRAM[Int](16)
        Foreach(16 by 1) { j => tmp(j) = 1}
        tmp
      }{_+_}
      out store a
    }

    val result = getMem(out)
    val gold = Array.tabulate(16){i => 10.to[Int]}
    printArray(gold, "expected: ")
    printArray(result, "result:   ")

    val cksum = gold.zip(result){_==_}.reduce{_&&_}
    println("PASS: " + cksum + " (Lab2Part2SimpleMemFold)")
  }
}


// FSM
object Lab2Part3BasicCondFSM extends SpatialApp { // Regression (Unit) // Args: none


  @virtualize
  def main() {
    val dram = DRAM[Int](32)
    Accel {
      val bram = SRAM[Int](32)
      val reg = Reg[Int](0)
      reg := 16
      FSM[Int]{state => state < 32} { state =>
        if (state < 16) {
          if (state < 8) {
            bram(31 - state) = state // 16:31 [7, 6, ... 0]
          } else {
            bram(31 - state) = state+1 // 16:31 [16, 15, ... 9]
          }
        }
        else {
          bram(state - 16) = if (state == 16) 17 else if (state == 17) reg.value else state // Test const, regread, and bound Mux1H
        }
      }{state => state + 1}

      dram(0::32) store bram
    }
    val result = getMem(dram)
    val gold = Array[Int](17, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                          29, 30, 31, 16, 15, 14, 13, 12, 11, 10, 9, 7, 6, 5, 4, 3, 2, 1, 0)
    printArray(result, "Result")
    printArray(gold, "Gold")
    val cksum = gold.zip(result){_ == _}.reduce{_&&_}
    println("PASS: " + cksum + " (Lab2Part3BasicCondFSM)")
  }
}



// FSMAlt
object Lab2Part3BasicCondFSMAlt extends SpatialApp {

  @virtualize
  def main() {
    val dram = DRAM[Int](32)
    Accel {
      val bram = SRAM[Int](32)
      val reg = Reg[Int](0)
      reg := 16
      FSM[Int]{state => state < 32} { state =>
        if (state < 8) {
            bram(state) = state 
        } else if (state < 16) {
            bram(state) = state*2 
        } else if (state < 24) {
            bram(state) = state*3
        } else {
          bram(state) = state*4
        }
      }{state => state + 1}

      dram(0::32) store bram      
    }
    val result = getMem(dram)
    val gold = Array[Int](0, 1, 2, 3, 4, 5, 6, 7, 16, 18, 20, 22, 24,
                          26, 28, 30, 48, 51, 54, 57, 60, 63, 66, 69,
                          96, 100, 104, 108, 112, 116, 120, 124)
    printArray(result, "Result")
    printArray(gold, "Gold")
    val cksum = gold.zip(result){_ == _}.reduce{_&&_}
    println("PASS: " + cksum + " (Lab2Part3BasicCondFSMAlt)")
  }
}


object Lab2Part4LUT extends SpatialApp {
  @virtualize
  def main() {
    type T = Int
    val M = 3
    val N = 3

    val in = ArgIn[T]
    val out = ArgOut[T]
    val i = ArgIn[T]
    val j = ArgIn[T]

    val input = args(0).to[T]
    val ind_i = args(1).to[T]
    val ind_j = args(2).to[T]

    setArg(in, input)
    setArg(i, ind_i)
    setArg(j, ind_j)

    Accel {
      // Your code here
      val inVal = in.value
      val iVal = i.value
      val jVal = j.value
      
      val lut = LUT[T](3,3)(1,2,3,4,5,6,7,8,9)
      val lut_ij = lut(iVal,jVal)
      out := inVal + lut_ij
    }

    val result = getArg(out)
    val goldArray = Array.tabulate(M * N){ i => i + 1 }
    val gold = input + goldArray(i*N + j)
    val pass = gold == result
    println("PASS: " + pass + "(Lab2Part4LUT)")
  }
}


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
    margin: FixPt[TRUE,_8,_8],
    digit: Int
  )

  @virtualize
  def main() {

    val outerPar = 1
    val midPar = 2
    val innerPar = 2

    type T = FixPt[TRUE,_4,_12]
    val tileM = 16
    val tileN = 16
    val tileK = 16

    // val picSize = 400.to[Int]
    // val numTrainImages = 60000.to[Int]
    // val numTestImages = 10000.to[Int]
    // val digits = 10.to[Int]
    val picSize = 4.to[Int]
    val numTrainImages = 2.to[Int]
    val numTestImages = 1.to[Int]
    val digits = 3.to[Int]


    // val M = ArgIn[Int]
    // val N = ArgIn[Int]
    // val K = ArgIn[Int]
    // setArg(M,args(0).to[Int])
    // setArg(N,args(1).to[Int])
    // setArg(K,args(2).to[Int])

    val rho = 0.5.to[T]
    //val rho = 0.04.to[T]
    val base_alpha = 0.1.to[T]
    //val base_alpha = 0.0001.to[T]

//    val train_data = (0::numTrainImages, 0::picSize){(i,j) => random[T](3)}
//    val train_labels = (0::1, 0::numTrainImages){(i,j) => 0.to[Int]}
//    val train_data = (0::numTrainImages, 0::picSize){(i,j) => 
//	mux(i==0,mux(j==0,0.5.to[T],mux(j==1,0.to[T],mux(j==2,1.to[T],0.2.to[T]))),mux(j==0,0.7.to[T],mux(j==1,0.3.to[T],mux(j==2,0.to[T],0.5.to[T]))))
//    }
//    val train_labels = Array[Int](1,0)
    val train_data = loadCSV2D[T]("images_train.csv",",")
    val train_labels = loadCSV2D[T]("labels_train.csv",",")
//    val test_data = (0::numTestImages, 0::picSize){(i,j) => random[T](3)}
//    val test_labels = (0::1, 0::numTestImages){(i,j) => 0.to[Int]}
//    val test_data = (0::numTestImages, 0::picSize){(i,j) => 
//	mux(j==0,0.to[T],mux(j==1,0.to[T],mux(j==2,0.9.to[T],0.2.to[T])))
//    }
//    val test_labels = Array[Int](1)
    val test_data = loadCSV2D[T]("images_test.csv",",")
    val train_labels = loadCSV2D[T]("labels_test.csv",",")
      
    val W_init = (0::digits, 0::picSize){(i,j) => 0.to[T]}
    val trainImages = DRAM[T](numTrainImages, picSize)
    val trainLabels = DRAM[Int](numTrainImages)
    val testImages = DRAM[T](numTestImages, picSize)
    val testLabels = DRAM[Int](numTestImages)
    val W = DRAM[T](digits, picSize)    

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
          val alpha = mux(i.to[Int] == label.value, 5*base_alpha, base_alpha)
          val ywx = Reg[T](0)
          Reduce(ywx)(picSize by 1) {j =>
            y.to[T] * W_sram(i,j) * img_sram(0,j)
          }{_ + _}
          val select = 1 - ywx.value
          val gk1_sram = SRAM[T](picSize)
          Foreach(picSize by 1){j =>
            gk1_sram(j) = mux(select > 0, rho * W_sram(i,j) - y.to[T] * img_sram(0,j),rho * W_sram(i,j))
          }
          Foreach(picSize by 1){j =>
            W_sram(i,j) = W_sram(i,j) - alpha * gk1_sram(j)
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
    printMatrix(accel_matrix, "Received: ")
    //printMatrix(gold_matrix, "Wanted: ")
    //val cksum = accel_matrix.zip(gold_matrix){_==_}.reduce{_&&_}
    //println("PASS: " + cksum + "(Lab2Part5GEMM)")
  }
}