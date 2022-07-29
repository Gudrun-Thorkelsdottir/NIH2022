#' Define the SpaCE object
#'
#' @slot input The input data.
#' @slot results The computed results.
#'
setClass("SpaCE",
  slots = c(
    input = "list",
    results = "list"
  )
)

#' Create an SpaCE object
#'
#' @title Data Input
#' @description Read ST data set to create an SpaCE object.
#' @param stPath Path to the folder.
#' @param platform Visium or oldST.
#' @return An SpaCE object
#' @details
#' Basically, `create.SpaCE.object` requires two parameters `seqPath` and `imagePath`, which point the standard output folders of 10x Visium data.
#'
#' The `seqPath` folder should include \cr
#' "barcodes.tsv.gz": spot level barcodes; \cr
#' "features.tsv.gz": list of genes; \cr
#' "matrix.mtx.gz": (sparse) matrix of counts.
#'
#' The `imagePath` folder should include \cr
#' âtissue_positions_list.csvâ : barcodes and spatial information; \cr
#' âtissue_lowres_image.pngâ : hematoxylin and eosin (H&E) image; \cr
#' âscalefactors_json.jsonâ : scaling factors for adjusting the coordinates .
#' @examples
#' SpaCE_obj_Visium <- create.SpaCE.object(stPath = "./inst/extdata/Visium_BC/", platform="Visium")
#' SpaCE_obj_oldST <- create.SpaCE.object(stPath = "./inst/extdata/oldST_BC23209_C1/", platform="oldST")
#' @rdname create.SpaCE.object
#' @export
#' @importFrom Matrix readMM
#' @importFrom jsonlite fromJSON
#'
create.SpaCE.object <- function(stPath,platform=c("Visium","oldST"))
{
  if(platform=="Visium")
  {
    st.matrix.data <- Matrix::readMM(paste0(stPath,"/filtered_feature_bc_matrix/matrix.mtx.gz"))
    st.matrix.data <- methods::as(st.matrix.data, "dgCMatrix")

    st.matrix.gene <- as.matrix(utils::read.csv(paste0(stPath,"/filtered_feature_bc_matrix/features.tsv.gz"),as.is=T,header=F,sep="\t"))
    st.matrix.anno <- as.matrix(utils::read.csv(paste0(stPath,"/filtered_feature_bc_matrix/barcodes.tsv.gz"),as.is=T,header=F,sep="\t"))

    rownames(st.matrix.data) <- st.matrix.gene[,2]
    colnames(st.matrix.data) <- st.matrix.anno[,1]

    st.matrix.data <- rm_duplicates(st.matrix.data)
    st.matrix.data <- rm_zeroRows(st.matrix.data)


    jsonFile <- jsonlite::fromJSON(paste0(stPath,"/spatial/scalefactors_json.json"))
    scalef <- jsonFile$tissue_lowres_scalef

    barcode <- utils::read.csv(paste0(stPath,"/spatial/tissue_positions_list.csv"),as.is=T,row.names=1,header=F)
    barcode[["comb"]] <- paste0(barcode[,2],"x",barcode[,3])
    barcode[["comb2"]] <- paste0(round(barcode[,4]*scalef,3),"x",round(barcode[,5]*scalef,3))
    barcode[["X"]] <- round(barcode[,4]*scalef,3)
    barcode[["Y"]] <- round(barcode[,5]*scalef,3)


    spotCoordinates <- barcode[colnames(st.matrix.data),c("X","Y")]
    rownames(spotCoordinates) <- barcode[colnames(st.matrix.data),c("comb")]

    colnames(st.matrix.data) <- rownames(spotCoordinates)

    imageFile <- paste0(stPath,"/spatial/tissue_lowres_image.png")


  }else{
    st.matrix.data <- as.matrix(read.table(paste0(stPath,"/counts.tsv")))
    colnames(st.matrix.data) <- gsub("X","",colnames(st.matrix.data))
    st.matrix.data <- rm_duplicates(st.matrix.data)
    st.matrix.data <- rm_zeroRows(st.matrix.data)

    spotCoordinates <- read.csv(paste0(stPath,"/spot_coordinates.csv"),row.names=1,header=TRUE)
    spotCoordinates <- spotCoordinates[colnames(st.matrix.data),]

    imageFile <- paste0(stPath,"/HE.jpg")
  }

  deconvolution <- as.matrix(read.csv(paste0(stPath,"/propMat_SpaCE.csv"),row.names=1))
  colnames(deconvolution) <- gsub("X","",colnames(deconvolution))

  deepLearning <- as.matrix(read.csv(paste0(stPath,"/propMat_deepLearning.csv"),row.names=1))
  colnames(deepLearning) <- gsub("X","",colnames(deepLearning))

  SpaCE_obj <- methods::new("SpaCE",
    input=list(
      counts=st.matrix.data,
      spotCoordinates=spotCoordinates,
      image=imageFile,
      platform=platform
    ),
    results=list(
      deconvolution=deconvolution,
      deepLearning=deepLearning
    )
  )

  SpaCE_obj
}


rm_duplicates <- function(mat){
  dupl <- duplicated(rownames(mat))
  if (sum(dupl) > 0){
    dupl_genes <- unique(rownames(mat)[dupl])
    mat_dupl <- mat[rownames(mat) %in% dupl_genes,,drop=F]
    mat_dupl_names <- rownames(mat_dupl)
    mat <- mat[!dupl,,drop=F]

    for(gene in dupl_genes){
      mat_dupl_gene <- mat_dupl[mat_dupl_names == gene,]
      dupl_sum <- apply(mat_dupl_gene,1,sum)
      max_flag <- which(dupl_sum==max(dupl_sum))
      mat[gene,] <- mat_dupl_gene[max_flag[1],] # in case two values are max
    }
  }
  return(mat)
}


rm_zeroRows <- function(mat){
  mat[rowSums(as.matrix(mat))>0,]
}

#' Visualization
#'
#' @title ST Visualization
#' @description FUNCTION_DESCRIPTION
#' @param visiualVector PARAM_DESCRIPTION
#' @param image PARAM_DESCRIPTION
#' @param platform PARAM_DESCRIPTION
#' @param colors PARAM_DESCRIPTION
#' @param limits PARAM_DESCRIPTION
#' @param titleName PARAM_DESCRIPTION
#' @param legendName PARAM_DESCRIPTION
#' @return OUTPUT_DESCRIPTION
#' @details DETAILS
#' @examples
#' NULL
#' @rdname visualSpatial
#' @export
#' @importFrom png readPNG
#' @importFrom grid rasterGrob unit
visualSpatial <- function(visiualVector,image,platform,colors,limits,titleName,legendName)
{
  library(ggplot2)

  if(platform=="Visium")
  {
    r <- png::readPNG(image)
    rg <- grid::rasterGrob(r, width=grid::unit(1,"npc"), height=grid::unit(1,"npc"))
    xDiml <- dim(r)[1] # dim pixel
    yDiml <- dim(r)[2] # dim pixel

    coordi <- t(matrix(as.numeric(unlist(strsplit(names(visiualVector),"x"))),nrow=2))
    fig.df <- data.frame(
      x=xDiml-coordi[,1],
      y=coordi[,2],
      value=visiualVector
    )
    rownames(fig.df) <- names(visiualVector)

    ggplot(fig.df,aes(x=x,y=y))+
      annotation_custom(rg) + # add background image
      geom_point(aes(colour=value))+
      scale_colour_gradientn(name=legendName, colours = colors,limits=limits)+
      scale_x_continuous(limits = c(0, xDiml), expand = c(0, 0)) +
      scale_y_continuous(limits = c(0, yDiml), expand = c(0, 0)) +
      ggtitle(titleName)+
      theme(
        plot.title = element_text(hjust = 0.5),
        axis.text = element_blank(),
        axis.title = element_blank(),
        axis.ticks = element_blank(),
        panel.grid = element_blank(),
        panel.border = element_blank()
      )+coord_flip()

  }else{
    r <- jpeg::readJPEG(image)
    rg <- grid::rasterGrob(r, width=grid::unit(1,"npc"), height=grid::unit(1,"npc"))
    pix_x <- dim(r)[2]
    pix_y <- dim(r)[1]

    coordi <- t(matrix(as.numeric(unlist(strsplit(names(visiualVector),"x"))),nrow=2))
    fig.df <- data.frame(
      x=coordi[,1],
      y=pix_y-coordi[,2],
      value=visiualVector
    )

    ggplot(fig.df,aes(x=x,y=y))+
      annotation_custom(rg)+  # add background image
      geom_point(aes(colour=value))+
      scale_colour_gradientn(name=legendName, colours = colors,limits=limits)+
      scale_x_continuous(limits = c(0, pix_x), expand = c(0, 0)) +
      scale_y_continuous(limits = c(0, pix_y), expand = c(0, 0)) +
      ggtitle(titleName)+
      theme_bw()+
      theme(
        plot.title = element_text(hjust = 0.5),
        axis.text = element_blank(),
        axis.title = element_blank(),
        axis.ticks = element_blank(),
        panel.grid = element_blank(),
        panel.border = element_blank()
      )

  }
}

#' @title Gene expression visualization
#' @description Visualize gene expression in ST data
#' @param SpaCE_obj An SpaCE object
#' @param gene Gene name
#' @param colors Legend color, Default: c("blue", "yellow", "red")
#' @param limits Value range
#' @return A ggplot2 object
#' @details DETAILS
#' @examples
#' SpaCE.visual.gene(SpaCE_obj_Visium,"EPCAM")
#' SpaCE.visual.gene(SpaCE_obj_oldST,"EPCAM")
#' @rdname SpaCE.visual.gene
#' @export
SpaCE.visual.gene <- function(
  SpaCE_obj,
  gene,
  colors = c("blue", "yellow", "red"),
  limits = NULL
)
{
  visiualVector <- log2(SpaCE_obj@input$counts[gene,]+1)
  names(visiualVector) <- paste0(SpaCE_obj@input$spotCoordinates[,"X"],"x",SpaCE_obj@input$spotCoordinates[,"Y"])

  visualSpatial(visiualVector,SpaCE_obj@input$image,SpaCE_obj@input$platform,colors,limits,gene,"Expr")
}

#' @title Cell type fraction visualization
#' @description Visualize cell type fraction in ST data
#' @param SpaCE_obj An SpaCE object
#' @param cellType Cell type
#' @param colors Legend color, Default: c("blue", "yellow", "red")
#' @param limits Value range
#' @return A ggplot2 object
#' @details DETAILS
#' @examples
#' SpaCE.visual.deconvolution(SpaCE_obj_Visium,"Malignant")
#' @rdname SpaCE.visual.deconvolution
#' @export
SpaCE.visual.deconvolution <- function(
    SpaCE_obj,
    cellType,
    colors = c("blue", "yellow", "red"),
    limits = c(0,1)
)
{
  visiualVector <- SpaCE_obj@results$deconvolution[cellType,]
  names(visiualVector) <- paste0(SpaCE_obj@input$spotCoordinates[,"X"],"x",SpaCE_obj@input$spotCoordinates[,"Y"])

  visualSpatial(visiualVector,SpaCE_obj@input$image,SpaCE_obj@input$platform,colors,limits,cellType,"Prop")
}

#' @title Cell type fraction visualization
#' @description Visualize cell type fraction in ST data
#' @param SpaCE_obj An SpaCE object
#' @param cellType Cell type
#' @param colors Legend color, Default: c("blue", "yellow", "red")
#' @param limits Value range
#' @return A ggplot2 object
#' @details DETAILS
#' @examples
#' SpaCE.visual.deepLearning(SpaCE_obj_Visium,"Malignant")
#' @rdname SpaCE.visual.deepLearning
#' @export
SpaCE.visual.deepLearning <- function(
    SpaCE_obj,
    cellType,
    colors = c("blue", "yellow", "red"),
    limits = c(0,1)
)
{
  visiualVector <- SpaCE_obj@results$deepLearning[cellType,]
  names(visiualVector) <- paste0(SpaCE_obj@input$spotCoordinates[,"X"],"x",SpaCE_obj@input$spotCoordinates[,"Y"])

  visualSpatial(visiualVector,SpaCE_obj@input$image,SpaCE_obj@input$platform,colors,limits,cellType,"Prop")
}
