// ghrd_10as066n2_ocm_0.v

// Generated using ACDS version 17.1 240

`timescale 1 ps / 1 ps
module ghrd_10as066n2_ocm_0 (
		input  wire        clk,        //   clk1.clk
		input  wire        reset,      // reset1.reset
		input  wire        reset_req,  //       .reset_req
		input  wire [17:0] address,    //     s1.address
		input  wire        clken,      //       .clken
		input  wire        chipselect, //       .chipselect
		input  wire        write,      //       .write
		output wire [7:0]  readdata,   //       .readdata
		input  wire [7:0]  writedata   //       .writedata
	);

	ghrd_10as066n2_ocm_0_altera_avalon_onchip_memory2_171_ehvj5ii ocm_0 (
		.clk        (clk),        //   input,   width = 1,   clk1.clk
		.address    (address),    //   input,  width = 18,     s1.address
		.clken      (clken),      //   input,   width = 1,       .clken
		.chipselect (chipselect), //   input,   width = 1,       .chipselect
		.write      (write),      //   input,   width = 1,       .write
		.readdata   (readdata),   //  output,   width = 8,       .readdata
		.writedata  (writedata),  //   input,   width = 8,       .writedata
		.reset      (reset),      //   input,   width = 1, reset1.reset
		.reset_req  (reset_req),  //   input,   width = 1,       .reset_req
		.freeze     (1'b0)        // (terminated),                     
	);

endmodule
