spiral_image <-
  function(
    img,
    invert = FALSE,
    size = 300,
    n_shades = 16,
    spiral_points = 5000,
    spiral_turns = 50,
    spiral_r0 = 0,
    spiral_r1_f = 1,
    thin = 0.00025,
    thick_f = 0.95,
    spiral_offset_angle = 0,
    col_line = "black",
    col_bg = "white"){
    
    # Read image --------------------------------------------------------------
    if(class(img) == "magick-image"){i <- img} else {i <- magick::image_read(img)}
    
    # Process image to sf polygon
    i_sf <-
      i |>
      magick::image_resize(paste0(size,"x",size,"^")) |>
      magick::image_crop(geometry = paste0(size,"x",size), gravity = "center") |>
      magick::image_convert(type = "grayscale") |>
      magick::image_quantize(max = n_shades, dither=FALSE) |>
      magick::image_flip() |> 
      magick::image_raster() |>
      dplyr::mutate(
        col2rgb(col) |> t() |> as_tibble(),
        col = scales::rescale(red, to = if(invert){c(0,1)}else{c(1, 0)})) |> 
      dplyr::select(-green, -blue, -red) |> 
      stars::st_as_stars() |>
      sf::st_as_sf(as_points = FALSE, merge = TRUE) |>
      sf::st_make_valid() |>
      sf::st_set_agr("constant") |> 
      sf::st_normalize()
    
    # Generate spiral ----------------------------------------------------------
    spiral <-
      spiral_coords(
        xo = 0.5,
        yo = 0.5,
        n_points = spiral_points,
        n_turns = spiral_turns,
        r0 = spiral_r0,
        r1 = 0.5 * spiral_r1_f,
        offset_angle = spiral_offset_angle) |>
      as.matrix() |>
      sf::st_linestring()
    
    # Compute the thick value
    thick <- ((((0.5*spiral_r1_f) - spiral_r0)/spiral_turns)/2)*thick_f
    
    intersections <-
      sf::st_intersection(i_sf, spiral) |>
      dplyr::mutate(n = scales::rescale(col, to=c(thin, thick))) |>
      dplyr::mutate(geometry = sf::st_buffer(geometry, n, endCapStyle = "ROUND")) |>
      sf::st_union()
    
    ggplot2::ggplot() + 
      ggplot2::geom_sf(data = intersections, fill = col_line, col = NA)+
      ggplot2::theme_void()+
      ggplot2::theme(panel.background = ggplot2::element_rect(fill = col_bg, colour = NA))+
      ggplot2::scale_x_continuous(limits = c(0,1))+
      ggplot2::scale_y_continuous(limits = c(0,1))
  }