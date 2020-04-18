<?php 

session_start();


if($_SESSION["loggedin_user"] == FALSE){
	header("Location: index.php");
  }


?>

<!DOCTYPE html>
<html>

<head>
    <title>
        Upload Image
    </title>
    <link rel="stylesheet" href="bootstrap/css/bootstrap.min.css">
    <link rel="stylesheet" href="css/upload_img.css">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    

</head>


<body class="container_login">
  
    <div class="container-fluid">
        
        <div class="row">
            <div class="col">
                <nav class="navbar navbar-expand-lg navbar-dark bg-dark sticky-top"> 
                   <a class="navbar-brand" href="#"> 
                       <img src="images/144315.svg" width="50" height="30" alt="">
                       <?php echo "".$_SESSION["loggedin_user"];?> 
                   </a>
                   <div class="collapse navbar-collapse" id="navbarText">
                     <ul class="navbar-nav mr-auto">
                        <li class="nav-item">
                          <a class="nav-link" href="#">Upload Image</a>
                        </li>
                        <li class="nav-item">
                          <a class="nav-link" href="previous_report/previous_report.php">Previous Report</a>
                        </li>
                    </ul>
                    <form class="inline-form" method="POST" action="php/logout.php">
                        <button class="btn btn-success my-2 my-sm-0" type="submit">Logout</button>
                    </form>
                    </div>
                </nav>
            </div>
        </div>
        <div class="row justify-content-center">
            <div class="wrap_login">
            <div class="col-lg-6 offset-lg-3">
                <img src="images/upload.png">
                <form action="php/upload_img.php" method="POST" enctype="multipart/form-data">
                        <input class="form-control-file" type="file" name="file">
                        <button class="btn btn-primary mt-4 btnmargin btn-group" type="submit">Submit</button> 
                </form>
            </div>
        </div>
        </div>
    


        
    </div>

</body>
</html>